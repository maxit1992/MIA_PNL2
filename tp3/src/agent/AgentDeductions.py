import ast
import csv

from .client.SingletonGroq import SingletonGroq


class AgentDeductions:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_DEDUCTION_PROMPT = sys_prompt = """Instructions:
- You are a helpful deductions assistant. You determine which users expenses apply to a tax reduction.
- Use the provided table that shows the applicable deduction categories with the maximum deductible amount.
- Always apply the general deduction category.
- If a user expense falls into a deductible category, apply the deductible amount declared by the user up to the maximum allowed for that category.
- Think step by step
- Return a response in the format {'thought': 'your line of thought', 'deductions': 'the deduction amounts with categories'} without additional text. 
    """

    def __init__(self, file_path):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq
        self.deductions_data = self.read_csv(file_path)

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
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                data.append(row)
        return data

    def answer(self, question: str) -> tuple[dict[str, str], tuple[int, int]]:
        """
        Decides which agents are involved in the user question and the question to be asked to the agents.

        Args:
            question (str): The user's question.
            agents (list): The agents name list.
        Returns:
            str: The output with the agents involved and the question to be asked to the agents.
        """
        sys_prompt = f"""{self.AGENT_DEDUCTION_PROMPT}

        Deductions data:
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
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            return answer, usage_tokens
        except (Exception,):
            return {"deductions": "I don\'t know"}, (0, 0)
