import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from util import get_response, get_vector_store

st.set_page_config(page_title='Chat With Websites', page_icon='🤖')

st.title(body='Chat With Websites')


with st.sidebar:
    st.title(body='Settings')
    web_URL = st.text_input(label='Enter a WebSite URL', placeholder='Website URL')
if web_URL is None or web_URL == '':
    st.info('Please enter a Website URL')
else:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi, I'm a bot. How can I help you")
        ]

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vector_store(web_URL=web_URL)

    user_input = st.chat_input(placeholder='Ask a question from Website')
    if user_input is not None and user_input != '':
        response_ = get_response(chat_history=st.session_state.chat_history, vector_store=st.session_state.vector_store, user_input=user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response_))


    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
