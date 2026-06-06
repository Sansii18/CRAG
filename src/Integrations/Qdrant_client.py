# The purpose of this phase is to create the knowledge base of the CRAG system. Instead of searching through all documents every time, document chunks are converted into embeddings and stored in Qdrant. Then similar chunks can be retrieved in milliseconds when a user asks a question.

# This file acts as a wrapper between the Python code and Qdrant Cloud. It manages all vector database operations.

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.config import get_config
from src.utils.logger import logger
from typing import List, Dict, Sequence
import uuid

class StoreQdrant:
    def __init__(self):
        """Initialize connection to Qdrant Cloud"""
        config = get_config('qdrant')
        try:
            self.client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                prefer_grpc=False  # Use HTTP for stability
            )
            self.collection_name = config.collection_name
            self.vector_size = config.vector_size
            logger.info("Connected to Qdrant Cloud successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise e
        
    def create_collection(self, force_recreate : bool = False):
        """Create Qdrant collection if it doesn't exist"""

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name in collection_names:
                if force_recreate:
                    logger.warning(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise e
        
    
    def upload_embeddings(self, chunks : List[str], embeddings : List[List[float]]):
        """Upload document chunks and embeddings to Qdrant"""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        
        try : 
            points = [
                PointStruct(
                    id = str(uuid.uuid4()),  # Generate unique ID for each point
                    vector = embedding,
                    payload = {"text": chunk}  # Store original text in payload
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]

            # Upload in batches (1000 points at a time)
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=list(batch)
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            logger.info(f"Uploaded {len(points)} vectors to Qdrant")

        except Exception as e:
            logger.error(f"Failed to upload embeddings: {e}")
            raise e
        
    
    def search(self, query_embedding : List[float], top_k : int = 5) -> List[Dict]:
        """Search Qdrant for similar documents"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            retrieved_chunks = [
                {
                    "text": result.payload["text"] if result.payload and "text" in result.payload else "",
                    "score": result.score,
                    "id": result.id
                }
                for result in results
            ]

            logger.info(f"Retrieved {len(retrieved_chunks)} documents")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e