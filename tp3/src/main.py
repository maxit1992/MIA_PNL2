"""
This script implements a Streamlit-based tax bot application. The application allows users ask how much tax they have
to pay next month based on their monthly income, deductions, and the current month. The bot uses a series of agents and
reasoning to provide an accurate response.
"""

import streamlit as st

from agent.AgentEnvironment import AgentEnvironment

# Start the Streamlit app
st.title("Tax Calculator Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    greeting = (
        "Hello I'm your tax calculator bot. Give me your monthly income, your deductions, the current month, how much "
        "you have already paid,  and I will calculate the tax amount to be paid next month.")
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
