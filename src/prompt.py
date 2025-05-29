

prompt_template = """


Use the information down below to answer the question user asks. 
If you do not know the answer, say that you dont know it but dont make up the answer on your own. 

Context: {context}
Question: {question}

Return the answer which is helpful and best matches the question asked by the user.

Answer: 

You must avoid any unnecessary repetition and try to keep the response to 2-3 sentences.
"""