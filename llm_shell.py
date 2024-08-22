
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

# run the llm
def run(llm, prompt, email, docs):
    result = llm.invoke(prompt.format(email=email, context=format_docs(docs)))
    return result

if __name__ == "__main__":

    load_dotenv()
    
    userPrompt = """
    Task: You are helping a user gain information on social services. Give effective responses.
    You must use a tool to answer questions. If you cannot use a tool, say "I don't know"
    """

    #grabbing the embeddings
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./.chromadb"
    )

    #initialized the llm model
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions = userPrompt)

    tools = [food_tool, housingshelter_tool, goods_tool, transit_tool, healthwellness_tool, money_tool, legal_tool, 
            dayservices_tool, specializedassistance_tool]
    
    agent = create_react_agent(llm, tools, base_prompt)

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
                print("generating response...")
                result = agent_executor.invoke({"input":line, "email": email, "context": context, "instructions": instructions })
                print(result["output"])
            else:
                break
        except Exception as e:
            print(e)

    # from flask import Flask, request, jsonify

    # app = Flask(__name__)

    # @app.route('/')
    # def hello():
    #     return 'Hello, World!'

    # @app.route('/chat', methods=['POST'])
    # def chat():
    #     email = 'test'
    #     context = "You are a LLM providing information about social services."
    #     instructions = "Use the appropriate tool to answer the user's question."
    #     print(request.json)
    #     line=request.json['question']
    #     result = agent_executor.invoke({"input":line, "email": email, "context": context, "instructions": instructions })
    #     print(result)
    #     return result

    # if __name__ == '__main__':
    #     app.run(host='0.0.0.0', port=6000, debug=True)



