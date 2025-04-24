import ast

from .client.SingletonGroq import SingletonGroq


class AgentAccountant:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_ACCOUNTANT_PROMPT = sys_prompt = """Instructions:
- You are a helpful accountant assistant, you helps the user calculating the tax amount to pay next month.
- Tax is annual, but paid monthly. The fiscal year runs from January to December.
- There are X steps to calculate the amount:
1- Project the monthly income to a year gross taxable amount.
2- Get the applicable deductions.
3- Apply the deductions to get the net taxable amount. If deductions are greater than the gross taxable amount, no tax has to be paid.
4- Get the fixed amount, the surplus base and the percentage over the surplus for the taxable amount.
6- Calculate the annual tax to be paid.
5- Proportionate the annual tax to the accumulated monthly tax.
6- Subtract the amount already paid from the the accumulated monthly tax and end the calculation here, returning the value.
- You have the help of three agents. The 'deductions' agent calculates the total deduction from the user's declaration. The 'calculator' agent do math. The 'percentage' agent return the applicable tax fixed amount and percentage over the surplus.
- To contact the 'deductions' agent, return a response in the format {'thought': 'your line of thought', 'deductions':'the deductions entered by the user with the category'}.
- To contact the 'calculator' agent, return a response in the format {'thought': 'your line of thought', 'calculator':'the calculation you need to solve'}
- To contact the 'percentage' agent, return a response in the format {'thought': 'your line of thought', 'percentage':'the annual projection after the deductions'}.
- For the final answer, return a response in the format {'thought': 'your line of thought', 'answer':'the tax amount to be paid next month'}.
- Think step by step.
- Be concise, don't add extra information. 
    """

    def __init__(self):
        """
        Initialize the class required services.
        """
        self.client = SingletonGroq().groq

    def answer(self, question: str, reasoning: list = None) -> tuple[dict[str, str], tuple[int, int]]:
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
        if reasoning:
            messages.extend(reasoning)
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=5000,
            temperature=0
        )
        try:
            usage_tokens = (chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
            answer = ast.literal_eval(chat_completion.choices[0].message.content)
            return answer, usage_tokens
        except (Exception,):
            return {'answer': 'I don\'t know'}, (0, 0)
