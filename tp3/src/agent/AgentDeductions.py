import ast

import csv

from src.client.SingletonGroq import SingletonGroq


class AgentDeductions:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_DEDUCTION_PROMPT = sys_prompt = """Instructions:
- You are a helpful deductions assistant that helps an accountant determine which expenses apply to a person's tax reduction.
- You are provided with a table showing the applicable deduction categories with the maximum deductible amount.
- The table has a general deduction category that must be always returned and special deductions (all the others).
- If a user expense falls into a deductible category, return the deductible amount declared by the user up to the maximum allowed for that category.
- Return a response in the format {'thought': 'your line of thought', 'deductions': 'the deduction amounts with categories'} without additional text. Respect the quotes. 
    """

    def __init__(self, file_path):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq
        self.deductions_data = self.read_csv(file_path)

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
        sys_prompt = f"""{self.AGENT_DEDUCTION_PROMPT}

        Context: 
        {self.deductions_data}"""
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
            return {'deductions':'I don\'t know'}