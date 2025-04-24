import ast

from src.client.SingletonGroq import SingletonGroq


class AgentAccountant:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_ACCOUNTANT_PROMPT = sys_prompt = """Instructions:
- You're a helpful accountant assistant which helps the user calculate how much they have to pay to the treasury in income tax in a fiscal year.
- The fiscal year runs from January to December.
- The tax is annual, but it's paid monthly. You have to calculate the annual income projection, deduct the deductions provided by the user, apply a base amount and an excedent tax percentage, and then project the resulting balance to the following month's accumulated balance. From this projection, subtract the amount already paid in the previous month to obtain the amount due for the following month.
- The user must provide their monthly income, potentially deductible expenses, the month to pay, and how much tax they have already paid. If any data is missing, return {'answer':'instructions for the user input'} without additional text.
- To accomplish this task, you have the help of three agents. The "deductions" agent gives you the actual applicable amounts to deduct based on the user's declaration. The "calculator" agent helps you do math. And the "percentage" agent helps you calculate the tax base value, the minimum and the percentage over the minimum to apply based on the annual projection.
- To contact the "deductions" agent, return a response in the format {'thought': 'your line of thought', 'deductions':'the deductions entered by the user with the category'} without additional text.
- To contact the "calculator" agent, return a response in the format {'thought': 'your line of thought', 'calculator':'the calculation you need to solve'} without additional text.
- To contact the "percentage" agent, return a response in the format {'thought': 'your line of thought', 'percentage':'the annual projection after the deductions'} without additional text.
- Let's think step by step, asking other agents, consuming their answers and continuing with the calculation.
- Whenever you have the answer, return a response in the format {'thought': 'your line of thought', 'answer':'the tax amount to be paid next month'} without additional text. 
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

    def answer(self, question: str, chat_history: list = None) -> dict[str, str]:
        """
        Decides which agents are involved in the user question and the question to be asked to the agents.

        Args:
            question (str): The user's question.
            agents (list): The agents name list.
        Returns:
            str: The output with the agents involved and the question to be asked to the agents.
        """
        messages = [{"role": "system", "content": self.AGENT_ACCOUNTANT_PROMPT},
                    {"role": "user", "content": question}]
        if chat_history:
            messages.extend(chat_history)
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=5000,
            temperature=0
        )
        try:
            return ast.literal_eval(chat_completion.choices[0].message.content)
        except (Exception,):
            return {}