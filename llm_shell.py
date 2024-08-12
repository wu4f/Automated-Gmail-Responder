
import re
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv
from langchain.tools import tool
import os
import json

# format the documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#TOOLS 
@tool 
def food_tool(parameter):
    """Useful to answer open ended questions about food by giving a list of food resources.
    Returns a list of food resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Food':
                entries.append(entry)

    return(entries)

@tool
def housingshelter_tool(parameter):
    """Useful to answer open ended questions about housing & shelter by giving a list of housing & shelter resources.
    Returns a list of housing & shelter resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Housing & Shelter':
                entries.append(entry)
    
    return entries

@tool
def goods_tool(parameter):
    """Useful to answer open ended questions about goods by giving a list of goods resources.
    Returns a list of goods resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Goods':
                entries.append(entry)
    
    return entries

@tool
def transit_tool(parameter):
    """Useful to answer open ended questions about transit by giving a list of transit resources.
    Returns a list of transit resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Transit':
                entries.append(entry)
    
    return entries

@tool
def healthwellness_tool(parameter):
    """Useful to answer open ended questions about health & wellness by giving a list of health & wellness resources.
    Returns a list of health & wellness resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Health & Wellness':
                entries.append(entry)
    
    return entries

@tool
def money_tool(parameter):
    """Useful to answer open ended questions about money by giving a list of money resources.
    Returns a list of money resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Money':
                entries.append(entry)
    
    return entries

@tool
def legal_tool(parameter):
    """Useful to answer open ended questions about legal by giving a list of legal resources.
    Returns a list of legal resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Legal':
                entries.append(entry)
    
    return entries

@tool
def dayservices_tool(parameter):
    """Useful to answer open ended questions about day services by giving a list of day services resources.
    Returns a list of day services resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Day Services':
                entries.append(entry)
    
    return entries

@tool
def specializedassistance_tool(parameter):
    """Useful to answer open ended questions about specialized assistance by giving a list of specialized assistance resources.
    Returns a list of specialized assistance resources"""
    entries = []
    
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Specialized Assistance':
                entries.append(entry)
    
    return entries

def chunking(documents):
    """Takes in Documents and splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    return chunks

def add_documents(vectorstore, chunks, n):
   for i in range(0, len(chunks), n):
       print(f"{i} of {len(chunks)}")
       vectorstore.add_documents(chunks[i:i+n])

@tool
def RAG_tool(parameter):
    "Useful to find specific information about a service including description, address, website, hours, and phone number."
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./.chromadb"
    )
    #JSONL LOADER
    loader = JSONLoader(
        file_path='./CategorizedData2.jsonl',
        jq_schema='.',
        text_content=False,
        json_lines=True
    )

    docs = loader.load()

    #add source to vectorstore
    chunks = chunking(docs)
    add_documents(vectorstore, chunks, 300)

    email = input("llm>> ")
    docs = vectorstore.similarity_search(email)
    context = "You are a LLM providing information about social services."

    # rag_chain = (
    #     {"context": context, "userPrompt": userPrompt, "email": email, "inpute_documents": docs}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # response = rag_chain.invoke(email)
    # return response

    prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "email", "userPrompt"])

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    instructions = "Use the appropriate tool to answer the user's question."

    response = chain.invoke(
        {"input_documents": docs, "email": email, "userPrompt": userPrompt, "instructions":instructions}, return_only_outputs=True
    )
    response["output_text"] = re.sub(r"\n", "<br>", response["output_text"])
    return {"response": response["output_text"]}

# run the llm
def run(llm, prompt, email, docs):
    result = llm.invoke(prompt.format(email=email, context=format_docs(docs)))
    return result

if __name__ == "__main__":

    load_dotenv()
    
    userPrompt = """
    Task: You are helping a user gain information on social services. Give effective responses.
    If you do not know the answer, please redirect the user to a related source.
    """

    #grabbing the embeddings
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chromadb"
    )

   #prints all the info in vectorstore
    print("RAG database initialized with the following sources.")
    retriever = vectorstore.as_retriever()
    docs = retriever.vectorstore.get()
    print(docs['metadatas'])
    print(docs['documents'])
    

    #initialized the llm model
    llm = ChatOpenAI(model='gpt-4o',temperature=0)
    # llm = GoogleGenerativeAI(
    #        model="gemini-1.5-pro",
    #        temperature=0)

    rag_prompt = '''

    Task: You are helping a user gain information on social services. Give effective responses.  
    If the user asks for a specific number of resources, pick out that number of resources in the requested category at random.
    If you do not know the answer, please redirect the user to a related source.

    Instructions: {instructions}
    Email: {email}
    Context: {context}

    '''
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(rag_prompt=rag_prompt)

    tools = [food_tool, housingshelter_tool, goods_tool, transit_tool, healthwellness_tool, money_tool, legal_tool, 
            dayservices_tool, specializedassistance_tool]
    
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) 

    print(f"Welcome to my application.  I am configured with these tools")
    for tool in agent_executor.tools:
        print(f'  Tool: {tool.name} = {tool.description}')

    while True:
        email = 'test'
        context = "You are a LLM providing information about social services."
        instructions = "Use the appropriate tool to answer the user's question."
        line = input("llm>> ")
        try:
            if line:
                result = agent_executor.invoke({"input":line, "email": email, "context": context, "instructions": instructions })
                print(result["output"])
            else:
                break
        except Exception as e:
            print(e)

    #chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    # while True:
    #     email = input("llm>> ")
    #     if email:
    #         docs = vectorstore.similarity_search(email)
    #         response = chain.invoke(
    #             {"input_documents": docs, "email": email, "userPrompt": userPrompt}, return_only_outputs=True
    #         )
    #         response["output_text"] = re.sub(r"\n", "<br>", response["output_text"])
    #         print(response["output_text"])
    #     else:
    #         break

