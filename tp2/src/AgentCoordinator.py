import ast

from SingletonGroq import SingletonGroq


class AgentCoordinator:
    """
    This class handles a single agent that decides which agents should be asked to retrieve information for the final
    answer.
    """
    AGENT_COORDINATOR_PROMPT = sys_prompt = """Instructions:
    - You're a helpful assistant who, based on a user's question, determines which CV candidates other agents should retrieve information from, so that another assistant answers the user's question with the proper context.
    - When asking to the CV candidate's agents, tell them they should answer about the candidate they have the CV of.
    - Be concise, do not add any unnecessary text.
    - Use the following format to return the text: {{'agents': ['candidate1', 'candidate2', ...], 'agents_prompt': 'the question to be asked to the candidates' agents'}}
    - If the question does not include the candidates names, return the name of the first candidate agent. 
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
        - Candidates names: {agents}"""
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
