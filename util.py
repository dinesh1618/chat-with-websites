from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

import os
from dotenv import load_dotenv
load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def load_URL(web_URL):
    loader = WebBaseLoader(web_path=web_URL)
    documents = loader.load()
    return documents


def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    text_splitter = text_splitter.split_documents(documents=documents)
    return text_splitter

def get_vectors(documents):
    vector_store = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
    return vector_store

def retriever_chain(chat_history, vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(messages=[
        MessagesPlaceholder(variable_name=chat_history),
        "user", "{input}",
        "user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"
    ])
    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retriever_chain

