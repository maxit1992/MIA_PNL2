import ast

from SingletonGroq import SingletonGroq


class AgentCoordinator:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_COORDINATOR_PROMPT = sys_prompt = """Instructions:
    - You are a helpful agent assistant that based on a user question, determines which agents should be asked to retrieve information for the final answer.
    - When asking other agents, tell them they should answer about the candidate they have the CV of.
    - Be concise, do not add any unnecessary text.
    - If you decide no agents are involved or if the agent is not specified, return the first agent in the list with the question to be asked.
    - Use the following format to return the text: {{'agents': ['agent1', 'agent2', ...], 'agents_prompt': 'the question to be asked to the agents'}}
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

    def answer(self, question: str, agents: [str]):
        """
        Decides which agents are involved in the user question and the question to be asked to the agents.

        Args:
            question (str): The user's question.
            agents (list): The agents name list.
        Returns:
            str: The output with the agents involved and the question to be asked to the agents.
        """
        sys_prompt = f"""{self.AGENT_COORDINATOR_PROMPT}
        - Involved agents: {agents}"""
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
        )
        try:
            result = ast.literal_eval(chat_completion.choices[0].message.content)
            if 'agents' in result and 'agents_prompt' in result:
                detected_agents = result.get('agents')
                agents_prompt = result.get('agents_prompt')
                return detected_agents, agents_prompt
            else:
                raise ValueError("Invalid response format")
        except (Exception,):
            return [agents[0]], question
