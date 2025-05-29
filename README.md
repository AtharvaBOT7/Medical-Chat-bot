Architectural Brief Overview:
1. We will collect some of the medical data in form of pdf files. We are using the Gale Encyclopedia of Medicine 3rd Edition as our medicinal pdf file.
2. Creating text chunks as the data is huge in size.
3. Creating the embeddings using these chunks, these are vector embeddings.
4. Now using these vector embeddings, we will create a semantic index. 
5. Now we will finally use this semantic index to build our knowledge base, to built the KB in our case, we will use Pinecone vector store.
6. The user will ask a query using the front end application. 
7. The user query will get converted to a query embedding and this embedding will be used to retrieve a ranked result from the knowledge base, the ranked result is the closest vector to the query embedding.
8. Now this ranked result is sent to our large language model, in our case we are using Meta Llama 2 model and then it will give us the exact answer. 

Tech stack used:
1. Python
2. Langchain
3. Hugging face
4. Flask 
5. Meta Llama 2
6. Pinecone

Virtual Environment used: medicalbot.

Steps to run:

1. python store_index.py
2. python app.py