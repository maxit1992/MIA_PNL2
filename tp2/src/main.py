"""
This script implements a Streamlit-based CV Chat Bot application. The application allows users to upload CVs, process
them to extract chunks of text, store the chunks in a vector database, and interact with the bot to ask questions about
the uploaded CVs.
"""

import os

import streamlit as st

from AgentEnvironment import AgentEnvironment

# Start the Streamlit app
st.title("CV Chat Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

if "cv1_file" not in st.session_state:
    uploaded_cv = st.file_uploader("Upload CV1", type=["pdf"], key=st.session_state["uploader_key"])
    if uploaded_cv:
        temp_file = f"resources/temp/{uploaded_cv.name}"
        with open(temp_file, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_cv.read())
        st.session_state['cv1_file'] = temp_file
        st.session_state["uploader_key"] += 1
        message = f"CV-1 {uploaded_cv.name} uploaded"
        st.session_state['messages'].append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)
        st.rerun()
elif "cv2_file" not in st.session_state:
    uploaded_cv = st.file_uploader("Upload CV2", type=["pdf"], key=st.session_state["uploader_key"])
    if uploaded_cv:
        temp_file = f"resources/temp/{uploaded_cv.name}"
        with open(temp_file, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_cv.read())
        st.session_state['cv2_file'] = temp_file
        st.session_state["uploader_key"] += 1
        message = f"CV-2 {uploaded_cv.name} uploaded"
        st.session_state['messages'].append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)
        st.rerun()
elif "cv3_file" not in st.session_state:
    uploaded_cv = st.file_uploader("Upload CV3", type=["pdf"], key=st.session_state["uploader_key"])
    if uploaded_cv:
        temp_file = f"resources/temp/{uploaded_cv.name}"
        with open(temp_file, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_cv.read())
        st.session_state['cv3_file'] = temp_file
        st.session_state["uploader_key"] += 1
        message = f"CV-3 {uploaded_cv.name} uploaded"
        st.session_state['messages'].append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)
        st.rerun()
elif "abot" not in st.session_state:
    st.session_state["abot"] = AgentEnvironment(st.session_state["cv1_file"],
                                                st.session_state["cv2_file"],
                                                st.session_state["cv3_file"])
    os.remove(st.session_state["cv1_file"])
    os.remove(st.session_state["cv2_file"])
    os.remove(st.session_state["cv3_file"])
else:
    question = st.chat_input("Ask a question:")
    if question:
        # Save the question
        st.session_state['messages'].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Answer the question
        abot = st.session_state["abot"]
        answer = abot.graph.invoke({"question": question})

        for message in answer['chat_history']:
            st.session_state['messages'].append(message)
            with st.chat_message(message['role']):
                st.markdown(message['content'])
