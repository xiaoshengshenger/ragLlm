from typing import List
import os
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

class ZhipuAiEmbeddings(Embeddings):
    def __init__(self):
        self.client = ZhipuAI()
        self.batch_size = 64

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        '''
        all_embeddings = []
        for i in range(0,len(texts),self.batch_size):
            input_embeddings = texts[i : i + self.batch_size]
            input_embeddings = [text.strip() for text in input_embeddings if text.strip()]
            print(len(texts))
            print(input_embeddings)
            response = self.client.embeddings.create(
                model="embedding-3",
                input=input_embeddings
            )
            batch_embeddings = [embeddings.embedding for embeddings in response.data]
        return all_embeddings.extend(batch_embeddings)
        '''
        response = self.client.embeddings.create(
                model="embedding-3",
                input=texts
            )
        return [embeddings.embedding for embeddings in response.data]
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]