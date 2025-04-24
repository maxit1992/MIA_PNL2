import ast

import csv

from src.client.SingletonGroq import SingletonGroq


class AgentPercentage:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_PERCENTAGE_PROMPT = sys_prompt = """Instructions:
- You are a helpful assistant who receives total taxable income and reviews a table to return the taxable base and the percentage of the excess over the applicable minimum.
- Return a response in the format {'percentage':'the applicable base, the minimum and the percentage'} without additional text. 
    """

    def __init__(self, file_path):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq
        self.tax_data = self.read_csv(file_path)

    def greetings(self):
        """
        Returns a greeting message from the agent.

        Returns:
            str: A greeting message from the agent.
        """
        return ("Hello, I am a Coordinator agent."
                " I decide which CVs should information be retrieved from to answer the user question.")

    @staticmethod
    def read_csv(file_path):
        """
        Reads a CSV file and returns its content as a list of dictionaries.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            list: A list of dictionaries where each dictionary represents a row in the CSV file.
        """
        data = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    def answer(self, question: str) -> dict[str, str]:
        """
        Decides which agents are involved in the user question and the question to be asked to the agents.

        Args:
            question (str): The user's question.
            agents (list): The agents name list.
        Returns:
            str: The output with the agents involved and the question to be asked to the agents.
        """
        sys_prompt = f"""{self.AGENT_PERCENTAGE_PROMPT}

        Context: 
        {self.tax_data}"""
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
            temperature=0
        )
        try:
            return ast.literal_eval(chat_completion.choices[0].message.content)
        except (Exception,):
            return {'percentage':'I don\'t know'}