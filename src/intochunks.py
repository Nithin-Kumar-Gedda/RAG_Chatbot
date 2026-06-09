from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class TextSpliter:
    """Split documnets into chunks for embedding."""

    def __init__(self, chunk_size: int=500, chunk_overlap:int=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", " ",""],
        )
        self.chunks =[]


    def split(self, docs:List) -> List:
        self.chunks = self.splitter.split_documents(docs)
        print(f"Split {len(docs)} documents into {len(self.chunks)} chunks")
        return self.chunks
    
        