# When documents are chunked in data_ingestion.py, it ends up with hundreds of plain text strings. Neither Qdrant nor any search algorithm can compare text directly, they need numbers. This file creates vector embeddings for the given data and stores it in vector databse (Qdrant).

import os 
from typing import List
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.utils.logger import logger

class GenerateEmbeddings:
    def __init__(self, model_name: str = "nvidia/nv-embedqa-e5-v5"):
        try:
            self.embeddings = NVIDIAEmbeddings(
                model=model_name,
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url="https://integrate.api.nvidia.com/v1",
                truncate="END" 
                # Document chunks are split so the most important context (topic, subject, key facts) appears at the start. Truncating from the end drops the least critical content and keeps the semantically meaningful part intact.
            )
            logger.info(f"Initialized NVIDIA NIM embeddings initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise e
        
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embeddings.embed_documents(chunks)
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise e
        
    def embed_query(self, query: str) -> List[float]:
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise e
        
