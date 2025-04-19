from SingletonGroq import SingletonGroq


class AgentLLM:
    """
    This class handles an agent that answers users questions based on information retrieved from other agents.
    """
    AGENT_LLM_PROMPT = sys_prompt = f"""Instructions:
    - You are a helpful agent assistant that analyzes answers to CV related questions retrieved from other agents that handles a single candidates, and answer a user question about all the candidates involved.
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the other agents context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response."""

    def __init__(self):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq

    def greetings(self):
        """
        Returns a greeting message from the agent.

        Returns:
            str: A greeting message from the agent.
        """
        return ("Hello, I am a principal LLM agent. I can answer questions about multiple candidates CVs,"
                " based on the answers retrieved from different CV agents.")

    def answer(self, question: str, context: {}):
        """
        Generates an answer to a user's question based on the provided context.

        Args:
            question (str): The user's question to be answered.
            context (dict): A dictionary containing context information.

        Returns:
            str: The generated answer to the question.
        """
        sys_prompt = f"""{self.AGENT_LLM_PROMPT}
                
        Context: 
        {context}"""
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
