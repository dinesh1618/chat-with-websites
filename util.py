from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def read_URL(web_url: str) -> list:
    loader = WebBaseLoader(web_path=web_url)
    documents = loader.load()
    return documents

def text_split(document: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents=document)
    return split_docs


def vector_store(split_docs:list):
    vector_store_ = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings())
    return vector_store_

def get_context_retriever_chain(retriever):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(messages=[
        ('system', "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}")
    ])
    stuff_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_doc_chain)

def get_vector_store(web_URL:str):
    docs = read_URL(web_url=web_URL)
    split_docs = text_split(document=docs)
    vector_store_ = vector_store(split_docs=split_docs)
    return vector_store_

def get_response(chat_history, vector_store, user_input):
    retriever = vector_store.as_retriever()
    retriever_chain = get_context_retriever_chain(retriever=retriever)
    conversation_chain = get_conversational_rag_chain(retriever=retriever_chain)
    response_ = conversation_chain.invoke({
        'chat_history': chat_history,
        "input": user_input
    })
    return response_['answer']

