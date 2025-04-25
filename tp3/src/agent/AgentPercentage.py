import ast
import csv

from .client.SingletonGroq import SingletonGroq


class AgentPercentage:
    """
    This class assists in getting the tax fixed amount, base surplus, and percentage over the surplus for a given
    taxable income. It reads tax data from a CSV file and uses a LLM to process the information.
    """
    AGENT_PERCENTAGE_PROMPT = sys_prompt = """Instructions:
- You are a helpful assistant. You receive a total taxable income and return a fixed tax amount, a base surplus and a percentage of the surplus.
- Use the provided table to get the values. Return the values for which the input is between the ranges.
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
        Reads a CSV file and returns its content.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            list: A list with the rows in the CSV file.
        """
        data = []
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                data.append(row)
        return data

    def answer(self, question: str) -> tuple[dict[str, str], tuple[int, int]]:
        """
        Returns the fixed amount, base surplus and percentage over the surplus for a given taxable income.

        Args:
            question (str): The question with the taxable amount.

        Returns:
            tuple: The tax values and the token usage. If the answer is not found, returns a default value.
        """
        sys_prompt = f"""{self.AGENT_PERCENTAGE_PROMPT}

        Tax data:
        {self.tax_data}"""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        try:
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            return answer, usage_tokens
        except (Exception,):
            return {'percentage': 'I don\'t know'}, (0, 0)
