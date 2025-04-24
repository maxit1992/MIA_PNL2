import ast
import csv

from .client.SingletonGroq import SingletonGroq


class AgentPercentage:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_PERCENTAGE_PROMPT = sys_prompt = """Instructions:
- You are a helpful assistant. You receive a total taxable income and return a fixed tax amount and a percentage of the surplus.
- Use the provided table to get the values.
- Return a response in the format {'percentage':'the fixed amount, the base surplus and the surplus percentage'} without additional text. 
    """

    def __init__(self, file_path):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq
        self.tax_data = self.read_csv(file_path)

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
        sys_prompt = f"""{self.AGENT_PERCENTAGE_PROMPT}

        Tax data:
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
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            return answer, usage_tokens
        except (Exception,):
            return {'percentage': 'I don\'t know'}, (0, 0)
