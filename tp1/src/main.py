"""
This script implements a Streamlit-based CV Chat Bot application. The application allows users to upload CVs, process
them to extract chunks of text, store the chunks in a vector database, and interact with the bot to ask questions about
the uploaded CVs.
"""

import os

import streamlit as st

from Chat import Chat
from TextProvider import TextProvider
from VectorDB import VectorDB

# Start the Streamlit app
st.title("CV Chat Bot")

# Initialize/Restore the Session
if "vectorDB" not in st.session_state:
    st.session_state["vectorDB"] = VectorDB()

if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = Chat()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

# Upload new CV
uploaded_cv = st.file_uploader("Upload CV", type=["pdf"], key=st.session_state["uploader_key"])
if uploaded_cv:
    temp_file = "resources/temp_cv.pdf"
    with open(temp_file, 'wb') as output_temporary_file:
        output_temporary_file.write(uploaded_cv.read())
    text_provider = TextProvider(temp_file)
    text = text_provider.get_chunks(chunk_max_size=512)
    vector_db = st.session_state.vectorDB
    vector_db.save_text(text)
    os.remove(temp_file)
    st.session_state["uploader_key"] += 1
    message = f"File {uploaded_cv.name} uploaded"
    st.session_state.messages.append({"role": "assistant", "content": message})
    with st.chat_message("assistant"):
        st.markdown(message)
    st.rerun()

# Answer users questions
question = st.chat_input("Ask a question:")
if question:
    # Save the question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Search For context
    vector_db = st.session_state.vectorDB
    context = vector_db.get_similar_text(question, top_k=2)
    chatbot = st.session_state.chatbot
    answer = chatbot.answer(question, context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
