from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader, CSVLoader


class DataLoader:   ## to read any kind of files...
    """Load documents from directory, supporting multiple file types."""

    supported_extension = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
    }
    def __init__(self, directory:str):
        self.directory = Path(directory)
        self.loaded_docs = []

    def load(self) -> List:
        """Load all documnets from directory"""
        self.loaded_docs = []

        for ext, loader_cls in self.supported_extension.items():
            files = list(self.directory.rglob(f"*{ext}"))
            print(f"Found {len(files)} {ext} files in the directory")

            for file in files:
                try:
                    loader = loader_cls(str(file))
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata['source_file'] = file.name
                        doc.metadata['file_type'] = ext.strip(".")

                    self.loaded_docs.extend(docs)

                except Exception as e:
                    print(f"Error loading {file.name}: {e}")

        print(f"\nTotal documents loaded: {len(self.loaded_docs)}")
        return self.loaded_docs
            

   