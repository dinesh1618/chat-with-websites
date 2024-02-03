import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from util import load_URL, text_split, get_vectors, retriever_chain


st.set_page_config(page_title='Get Info from WebSites.', page_icon='🤖')

st.title(body="Ask Anything from uploaded Website.")


def get_response(user_input):
    return 'I do not know'

if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content='Hi, I am bot. How may I can help you...')
    ]

with st.sidebar:
    website_URL = st.text_input(label='Settings', placeholder='Provide a WebSite URL')


if website_URL is None or website_URL == '':
    st.info("Please enter a URL")
else:
    user_query = st.chat_input(placeholder='Ask anything about provided WebSite')
    documents = load_URL(web_URL=website_URL)
    data = text_split(documents=documents)
    vector_store = get_vectors(documents=data)
    if user_query is not None and user_query != '':
        response_ = get_response(user_input=user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response_))
    st.write(retriever_chain(chat_history=st.session_state.chat_history, vector_store=vector_store))
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message(name='AI'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message(name='Human'):
                st.write(message.content)
