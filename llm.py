import re
from typing import Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class QueryModel(BaseModel):
    email: str
    userPrompt: Union[str, None] = None

# Format the documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Run the llm
def run(llm, prompt, email, docs):
    result = llm.invoke(prompt.format(email=email, context=format_docs(docs)))
    return result

@app.get("/")
def aerllm(q: Union[str, None] = None, userPrompt: Union[str, None] = None):
    email: str = None
    # Loading environment variables
    load_dotenv()

    # Grabbing the embeddings
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chromadb"
    )
    # Initialize the llm model
    llm = ChatOpenAI(model='gpt-4o',temperature=0)
    # llm = GoogleGenerativeAI(
    #        model="gemini-1.5-pro",
    #        temperature=0)

    if q is not None:
        email = q
    else:
        raise ValueError("No valid question given")
    docs = vectorstore.similarity_search(email)
    rag_prompt = '''

    Task: You are helping a user gain information on social services. Give effecient responses.
    If you do not know the answer, return 'I don't know'. 
        
    Email: {email}
    Context: {context}
    Additional Guidelines: {userPrompt}
    '''

    prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "email", "userPrompt"])

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    response = chain.invoke(
        {"input_documents": docs, "email": email, "userPrompt": userPrompt}, return_only_outputs=True
    )
    response["output_text"] = re.sub(r"\n", "<br>", response["output_text"])
    return {"response": response["output_text"]}

@app.post("/submit-query")
async def submit_query(query: QueryModel):
    # Loading environment variables
    load_dotenv()

    # Grabbing the embeddings
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chromadb"
    )
    # Initialize the llm model
    llm = ChatOpenAI(model='gpt-4o',temperature=0)
    # llm = GoogleGenerativeAI(
    #        model="gemini-1.5-pro",
    #        temperature=0)

    docs = vectorstore.similarity_search(query.email)
    rag_prompt = '''
    
    Task: You are helping a user gain information on social services. Give effecient responses.
    If you do not know the answer, return 'I don't know'. 
    
    Email: {email}
    Context: {context}
    Additional Guidelines: {userPrompt}
    '''

    prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "email", "userPrompt"])

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    response = chain.invoke(
        {"input_documents": docs, "email": query.email, "userPrompt": query.userPrompt}, return_only_outputs=True
    )
    response["output_text"] = re.sub(r"\n", "<br>", response["output_text"])
    return {"response": response["output_text"]}




if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
