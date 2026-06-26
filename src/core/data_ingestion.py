# This file is created because the raw data / pdf file cannot be directly used.
# This file creates vector embeddings for the given data and stores it in vector databse (Qdrant). 

import os 
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import logger
from langchain.schema import Document

class DataIngestion:
    def __init__(self, Chunk_Size : int = 500, Chunk_Overlap : int = 50):
        self.chunk_size = Chunk_Size
        self.chunk_overlap = Chunk_Overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_data(self, directory) -> List[Document]:
        docs = []
        # PDF LOADER 
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader 
        )
        docs.extend(pdf_loader.load())
        logger.info(f"Loaded {len(docs)} PDFs from {directory}")
    
    # LOAD TEXT FILES
        text_loader = DirectoryLoader(
                directory,
                glob="**/*.txt",
                loader_cls=TextLoader
        )
        docs.extend(text_loader.load())
        logger.info(f"Loaded {len(docs)} text files from {directory}")
        return docs
    

    def split_data(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        Chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(Chunks)} chunks")
        return Chunks
    
    def process_directory(self, directory): 
        if not Path(directory).exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        documents = self.load_data(directory)
        Chunks = self.split_data(documents)

        return Chunks
    
# RUN THIS FILE FROM BASE DIRECTORY WITH THIS COMMAND "python -m src.core.data_ingestion"

# TO TEST : 
# print("Hello from data ingestion module")