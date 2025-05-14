# Initial Bash Command:
# pip install langchain==0.1.1 langchain-core langchain-community==0.0.13 pinecone-client==3.0.0 pinecone-datasets==0.7.0 openai==0.27.7 tiktoken==0.4.0 python-dotenv beautifulsoup4 requests feedparser Flask

import os
from dotenv import load_dotenv
import time
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API Keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name and embedding dimension
index_name = "jobhuntbot"
embedding_dimension = 1536

# Check if the index exists, create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"Pinecone index '{index_name}' created and ready.")
else:
    print(f"Pinecone index '{index_name}' already exists.")

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load and process documents
data_path = "/content/data/"  # Ensure this path is correct
loader = DirectoryLoader(
    data_path,
    glob="./*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1300,
    chunk_overlap=800,
)
texts = text_splitter.split_documents(documents)
print(f"Created {len(texts)} chunks from {len(documents)} documents.")

# Add chunks to Pinecone
print("Adding text chunks to Pinecone...")
vector_store.add_documents(texts)  
print(f"{len(texts)} text chunks added to Pinecone index '{index_name}'.")

# Query Pinecone (For Testing - Consider removing or adapting for job search)
# Note: The namespace 'ns1' was hardcoded in your original query.
# You'll likely want to remove or adjust this for your job search logic.
# Also, the filter for 'genre' is specific to your test data and won't be relevant
# for job descriptions unless you add such metadata.
# response = index.query(
#     namespace="ns1",
#     vector=[0.1, 0.3],
#     top_k=2,
#     include_values=True,
#     include_metadata=True,
#     filter={"genre": {"$eq": "action"}}
# )
# print("\nInitial Pinecone Query Response:")
# print(response)