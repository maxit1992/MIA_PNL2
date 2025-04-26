import ast
import csv

from .client.SingletonGroq import SingletonGroq


class AgentDeductions:
    """
    This class performs the role of a deductions assistant, helping to determine which expenses apply to a tax
    deduction. It reads the deducible categories from a CSV file and uses a LLM with reasoning to process the
    information.
    """
    AGENT_DEDUCTION_PROMPT = sys_prompt = """Instructions:
- You are a helpful deductions assistant. You determine which users expenses apply to a tax reduction.
- Use the provided table that shows the applicable deduction categories with the maximum deductible amount.
- Always apply the general deduction category. 
- If a user expense does not fall into a non general category, skip it.
- If a user expense falls into a not general category deductible category, apply the deductible amount declared by the user up to the maximum allowed for that category. 
- Think step by step
- Return a response in the format {'thought': 'your line of thought', 'deductions': 'Applicable deductions: the deduction amounts with categories'} without additional text. 
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
        Returns the applicable deductible categories with the deductible amount based on the input.

        Args:
            question (str): The question with the declared deductions.

        Returns:
            tuple: The deduction values and the token usage. If the answer is not found, returns a default value.
        """
        sys_prompt = f"""{self.AGENT_DEDUCTION_PROMPT}

        Deductions data:
        {self.deductions_data}"""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
            model="llama-3.3-70b-versatile"
        )
        try:
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            return answer, usage_tokens
        except (Exception,):
            return {"deductions": "I don\'t know"}, (0, 0)
