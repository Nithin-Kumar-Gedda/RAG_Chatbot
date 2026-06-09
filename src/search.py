from typing import List, Tuple, Dict, Any
from src.vectorstore import VectorStore
from src.embeddings import EmbeddingManager



class RagRetriever:

    def __init__(self,vectorstore:VectorStore, embedding_manager: EmbeddingManager):
        """ 
        Initialize the retriever
        
        Args:
            vectorstore: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vectorstore= vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k:int=5, score_threshold: float =0.0) -> List[Dict[str,Any]]:
        """ 
        retrieve relevant documents for a query
        
        args:
            query: the search query
            top_k: Number of top result to return
            score_threshold: minimum similarity score threshold
            
        returns:
            list of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top-K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_emb = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vectorstore.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k
            ) 

            # process result
            retrieved_docs =[]

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # convert the distance to similarity score
                    similarity_score = 1- distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score':similarity_score,
                            'distance': distance,
                            'rank':i+1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents")

            else:
                print(f"No documents found")

            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
