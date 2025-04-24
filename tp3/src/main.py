"""
This script implements a Streamlit-based CV agent bot application. The application allows users to upload up to 3 CVs,
process them to extract chunks of text, store the chunks in a vector database, and interact with agent bots that
coordinate themselves to answer questions about the uploaded CVs.
"""

import streamlit as st

from agent.AgentEnvironment import AgentEnvironment

# Start the Streamlit app
st.title("Tax Calculator Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    greeting = (
        "Hello I'm your tax calculator bot. Give me your monthly income, your deductions, the current month, and I will"
        " calculate the tax amount to be paid next month.")
    st.session_state['messages'].append({"role": "assistant", "content": greeting})
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Input the data:")

if "tax_bot" not in st.session_state:
    st.session_state["tax_bot"] = AgentEnvironment()
else:
    if question:
        # Answer questions
        st.session_state['messages'].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        tax_bot = st.session_state["tax_bot"]
        agent_state = tax_bot.graph.invoke({"question": question})
        for message in agent_state['chat_history']:
            st.session_state['messages'].append(message)
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        tokens_usage = (f"Usage: input tokens={agent_state['input_tokens']}, "
                        f"output tokens={agent_state['output_tokens']}, "
                        f"reasoning tokens={agent_state['reasoning_tokens']}")
        st.session_state['messages'].append({"role": "assistant", "content": tokens_usage})
        with st.chat_message("assistant"):
            st.markdown(tokens_usage)
        answer = agent_state['answer']
        st.session_state['messages'].append({"role": "assistant", "content": f"You have to paid {answer}"})
        with st.chat_message("assistant"):
            st.markdown(answer)
