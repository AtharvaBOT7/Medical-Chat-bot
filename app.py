from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
from langchain.schema import Document, BaseRetriever
from typing import List, Any
import asyncio
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chat-bot"

index = pc.Index(index_name)

embeddings = download_hugging_face_embeddings()

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

chain_type_kwargs = {"prompt":prompt}

llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  verbose=True,
                  config={'max_new_tokens':512,
                          'temperature':0.3})


class PineconeRetriever(BaseRetriever):
    embedding_model: Any
    index: Any
    k: int = 2

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        results = self.index.query(vector=query_vector, top_k=self.k, include_metadata=True)

        return [
            Document(page_content=match["metadata"]["text"], metadata={"score": match["score"]})
            for match in results["matches"]
        ]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_relevant_documents, query)

retriever = PineconeRetriever(embedding_model=embeddings, index=index, k=2)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    retriever=retriever
)

@app.route("/")
def index():
    return render_template('chat.html')



if __name__ == '__main__':
    app.run(debug=True)