# This file runs all the 3 files (data_ingestion.py , VectorEmbeddings.py and Qdrant_client.py) in sequence. It is the main entry point for preparing the data and uploading it to Qdrant. It can be run as a standalone script or imported as a module.

import sys
import argparse
sys.path.insert(0, '.')
from src.core.data_ingestion import DataIngestion
from src.core.VectorEmbeddings import GenerateEmbeddings
from src.Integrations.Qdrant_client import StoreQdrant
from src.utils.logger import logger

def main(data_dir : str = "data/raw"):
    logger.info("Starting data preparation and upload process.")
    logger.info(f"Data directory: {data_dir}")

    # INGEST THE DOCUMENTS 
    logger.info("Ingesting documents...")
    ingestor = DataIngestion()
    chunks = ingestor.process_directory(data_dir)
    logger.info(f"Total chunks created: {len(chunks)}")

    # GENERATE EMBEDDINGS
    logger.info("Generating embeddings...")
    embedder = GenerateEmbeddings()
    embeddings = embedder.embed_chunks(
        [chunk.page_content for chunk in chunks]
    )

    # UPLOAD TO QDRANT
    qdrant = StoreQdrant()
    qdrant.create_collection(force_recreate=False)
    # force_recreate = False , avoids creating the qdrant again if it already exists, which is useful for testing and development. In production, it is set to True to ensure a clean slate.
    qdrant.upload_embeddings(
        [chunk.page_content for chunk in chunks],
        embeddings
    )

    logger.info("Data preparation and upload process completed successfully.")
    logger.info(f"{len(chunks)} chunks are ready for retrieval.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and upload documents to Qdrant.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw documents to process."
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir)