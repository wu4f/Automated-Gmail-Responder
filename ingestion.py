import json
import re
import os
import requests
import unidecode
import itertools
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
import urllib3

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader

def clean_text(text):
    """Replaces unicode characters and strips extra whitespace"""
    text = unidecode.unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_documents(documents):
    """Cleans page_content text of Documents list"""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents

def scrape_articles(links):
    """Scrapes list of links, extracts article text, returns Documents"""
    # Scrape list of links
    loader = AsyncHtmlLoader(links)
    docs = loader.load()
    # Extract article tag
    transformer = BeautifulSoupTransformer()
    docs_tr = transformer.transform_documents(
        documents=docs, tags_to_extract=["article"]
    )
    clean_documents(docs_tr)
    return docs_tr

def load_pdf_documents(dir):
    """Loads all PDFs in given directory"""
    loader = PyPDFDirectoryLoader(dir)
    docs = loader.load()
    return docs

def chunking(documents):
    """Takes in Documents and splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    return chunks

def extract_text(html):
    """Used by loader to extract text from div tag with id of main"""
    soup = BeautifulSoup(html, "html.parser")
    div_main = soup.find("div", {"class": "main-content"})
    if div_main:
        return div_main.get_text(" ", strip=True)
    return " ".join(soup.stripped_strings)

def scrape_main(url, depth):
    """Recursively scrapes URL and returns Documents"""
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=depth,
        timeout=20,
        use_async=True,
        prevent_outside=True,
        check_response_status=True,
        continue_on_failure=True,
        extractor=extract_text,
    )
    docs = loader.load()
    clean_documents(docs)
    return docs

def load_config(filename):
    """Reads configuration from a JSON file"""
    with open(filename, "r") as f:
        config = json.load(f)
    return config

def add_documents(vectorstore, chunks, n):
   for i in range(0, len(chunks), n):
       print(f"{i} of {len(chunks)}")
       vectorstore.add_documents(chunks[i:i+n])

def load_document(loader_class, website_url):
    """
    Load a document using the specified loader class and website URL.

    Args:
    loader_class (class): The class of the loader to be used.
    website_url (str): The URL of the website from which to load the document.

    Returns:
    str: The loaded document.
    """
    loader = loader_class([website_url])
    return loader.load()

#  Methods to filter entries
def populate_data(input_file, output_file):
    categories = ["Food", "Housing & Shelter", "Goods", "Transit", "Health & Wellness", "Money", 
                  "Care & Safety", "Work", "Legal", "Day Services", "Specialized Assistance"]
    
    for i in range(len(categories)):
        entries = []
        with open(input_file, 'r') as infile:
            for line in infile:
                entry = json.loads(line.strip())
                if entry.get('general_category') == categories[i]:
                    entries.append(entry)
        
        with open(output_file, 'a') as outfile:
            #outfile.write(f'The services that provide {categories[i]} are: \n')
            for entry in entries:
                json.dump(entry, outfile)
                outfile.write('\n')
        
input_file = 'data.jsonl'
output_file = 'CategorizedData2.jsonl'

#populate_data(input_file, output_file)

if __name__ == "__main__":
    # loading environment variables
    load_dotenv()

    # Check if OpenAI API key is provided in environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        # Prompt the user toenter the OpenAI API key if not found
        openai_api_key = input("Please enter your OpenAI API key: ")
        # Set the environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    # Load configuration
    #config = load_config("config.json")
    #bulletin_websites = config["bulletin_websites"]
    #cs_website = config["cs_website"]
    #blanchet = config["blanchet"]
    #transition_project = config["transition project"]
    
    #cs_courses_websites = config["cs_courses_websites"]
    urllib3.disable_warnings()

    # Initialize vectorstore
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./.chromadb"
    )
    
    response = requests.get("https://rosecityresource.streetroots.org/api/query")
    data = response.json() #make json file
    with open('data.jsonl', 'w') as file: #makes new data.jsonl file
        for i in range(len(data)):
            json_string = json.dumps(data[i])
            file.write(json_string + '\n')

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
    print("documents loaded into vectorstore")

