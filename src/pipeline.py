from src.data_loader import DataLoader
from src.intochunks import TextSpliter
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RagRetriever
from src.generator import ResponseGenerator

def build_pipeline(data_directory:str) -> RagRetriever:

    # STEP 1
    loader = DataLoader(data_directory)
    docs = loader.load()

    # STEP 2
    splitter = TextSpliter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split(docs)

    # STEP 3
    embeddings_mng = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_mng.generate_embeddings(texts)

    # STEP 4
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks,embeddings)

    # STEP 5
    retriever = RagRetriever(vectorstore,embeddings_mng)
    generator = ResponseGenerator()

    return retriever, generator


def ask(retriever: RagRetriever, generator: ResponseGenerator, question: str):
    print(f"\nQuestion: {question}")

    docs = retriever.retrieve(question, top_k=2)

    answer = generator.generate(question,docs)

    print(f"Answer: {answer}")

    return answer


