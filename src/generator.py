from typing import Dict,List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
groq_api_key = os.environ['GROQ_API_KEY']

class ResponseGenerator:


    def __init__(self, model_name = "llama-3.1-8b-instant"):
        self.model = ChatGroq(groq_api_key=groq_api_key, model=model_name, temperature=0.1, max_tokens=1024)

    def generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        context = "\n\n".join([doc['content'] for doc in retrieved_docs]) if retrieved_docs else ""

        if not retrieved_docs:
            return "I coudn't find any relevant information."
            
        #generate the answer from GROQ LLM

        prompt=f"""Use the following context to answer the question concisely.
        context: 
        {context}
        question:
        {query}"
        Answer:"""

        response = self.model.invoke([prompt.format(context=context, query=query)])
        return response.content
            
            
        