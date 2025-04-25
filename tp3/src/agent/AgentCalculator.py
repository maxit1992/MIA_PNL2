import ast

from .client.SingletonGroq import SingletonGroq


class AgentCalculator:
    """
    This class performs the role of a calculator agent, helping with mathematical calculations.
    It uses a LLM to process the formula in natural language.
    """
    AGENT_CALCULATOR_PROMPT = """Instructions:
- You are a helpful calculator. You execute math.
- Think step by step.
- Return a response in the format {"thought": "your line of thought", "calculator":"the mathematical calculation executable by python eval"} without additional text. 
    """

    def __init__(self):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq

    def answer(self, question: str) -> tuple[dict[str, str], tuple[int, int]]:
        """
        Answers the mathematical question.

        Args:
            question (str): The user's question.

        Returns:
            tuple: The calculation result and the token usage. If the answer is not found, returns a default value.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "system", "content": self.AGENT_CALCULATOR_PROMPT},
                      {"role": "user", "content": question}],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        try:
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            answer['calculator'] = eval(answer['calculator'])
            return answer, usage_tokens
        except (Exception,):
            return {'calculator': 'I don\'t know'}, (0, 0)
