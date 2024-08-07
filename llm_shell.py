#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
#from langchain_openai import OpenAIEmbeddings,ChatOpenAI
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os



# format the documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# run the llm
def run(llm, prompt, email, docs):
    result = llm.invoke(prompt.format(email=email, context=format_docs(docs)))
    return result

if __name__ == "__main__":

    load_dotenv()
    
    userPrompt = """
    Task: You are helping a user gain information on social services. Write a short, concise response to their question. 
    """

     #grabbing the embeddings
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chromadb"
    )

    print("ALL RESOURCES HERE")
    print("RAG database initialized with the following sources.")
    retriever = vectorstore.as_retriever()
    docs = retriever.vectorstore.get()
    print(docs['metadatas'])
    print(docs['documents'])

    #initialized the llm model
    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)

    rag_prompt = '''
    Task: You are helping a user gain information on social services. Write a short, concise response to their question. 
    
    Email: {email}
    Context: {context}
    Additional Guidelines: {userPrompt}
    '''

    prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "email"])

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    while True:
        email = input(">> ")
        if email:
            docs = vectorstore.similarity_search(email)
            response = chain.invoke(
                {"input_documents": docs, "email": email, "userPrompt": userPrompt}, return_only_outputs=True
            )
            #response["output_text"] = re.sub(r"\n", "<br>", response["output_text"])
            print(response["output_text"])
        else:
            break

