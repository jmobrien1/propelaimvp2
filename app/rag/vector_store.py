import os
from typing import List
from fastapi import HTTPException

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# Vector Store Manager
class VectorStoreManager:
    """Manages the vector store for RAG operations"""

    def __init__(self):
        self.vector_store_path = "compliance_vector_store.faiss"
        self.embeddings = None
        self.vector_store = None

    def initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500,
                                detail="OPENAI_API_KEY not configured")
        self.embeddings = OpenAIEmbeddings()

    def create_chunks(self, text_content: str, filename: str,
                      doc_category: str) -> List[Document]:
        """Create document chunks for vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""])

        chunks = text_splitter.split_text(text_content)
        documents = []

        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk,
                           metadata={
                               "source": filename,
                               "chunk_id": i,
                               "document_category": doc_category,
                               "content_length": len(chunk)
                           })
            documents.append(doc)

        return documents

    def add_to_vector_store(self, documents: List[Document]):
        """Add documents to vector store"""
        if not self.embeddings:
            self.initialize_embeddings()

        if self.vector_store is None:
            # Load existing or create new
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True)
            else:
                self.vector_store = FAISS.from_documents(
                    documents, self.embeddings)
                return

        # Add new documents
        if documents:
            self.vector_store.add_documents(documents)

    def save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self):
        """Load vector store from disk"""
        if not self.embeddings:
            self.initialize_embeddings()

        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True)
            return True
        return False
