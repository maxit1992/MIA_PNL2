import ast

import csv

from src.client.SingletonGroq import SingletonGroq


class AgentCalculator:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_CALCULATOR_PROMPT = """Instructions:
- You are a helpful calculator assistant that helps doing math.
- Be precise and think step by step.
- Return a response in the format {"thoughts": "your line of thoughts", "calculator":"the calc answer"} without additional text. 
    """

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
        return ("Hello, I am a Coordinator agent."
                " I decide which CVs should information be retrieved from to answer the user question.")

    def answer(self, question: str) -> dict[str, str]:
        """
        Decides which agents are involved in the user question and the question to be asked to the agents.

        Args:
            question (str): The user's question.
            agents (list): The agents name list.
        Returns:
            str: The output with the agents involved and the question to be asked to the agents.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.AGENT_CALCULATOR_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        try:
            return ast.literal_eval(chat_completion.choices[0].message.content)
        except (Exception,):
            return {'calculator':'I don\'t know'}