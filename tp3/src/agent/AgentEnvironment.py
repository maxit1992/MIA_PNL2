from typing import TypedDict

from langgraph.graph import StateGraph, END

from .AgentAccountant import AgentAccountant
from .AgentCalculator import AgentCalculator
from .AgentDeductions import AgentDeductions
from .AgentPercentage import AgentPercentage


class AgentState(TypedDict):
    question: str
    reasoning: list[dict[str, str]]
    chat_history: list[dict[str, str]]
    answer: str
    next_agent: str
    agent_prompt: str
    token_usage_history: list[tuple[int, int]]
    input_tokens: int
    reasoning_tokens: int
    output_tokens: int


class AgentEnvironment:
    """
    This class manages the environment for coordinating multiple agents to answer user questions based on CV data.
    It initializes the agents, sets up a state graph, and orchestrates the flow of information between agents.
    """

    def __init__(self):
        """
        Initializes the AgentEnvironment with the provided CV files, sets up agents, and compiles the state graph.
        """
        # Initialize the agents
        self.accountant = AgentAccountant()
        self.calculator = AgentCalculator()
        self.deductions = AgentDeductions("resources/deductions_data.csv")
        self.percentage = AgentPercentage("resources/tax_data.csv")

        # Initialize the state graph
        graph = StateGraph(AgentState)
        graph.add_node("accountant", self.process_accountant)
        graph.add_node("calculator", self.process_calculator)
        graph.add_node("deductions", self.process_deductions)
        graph.add_node("percentage", self.process_percentage)
        graph.add_node("pricer", self.pricer)
        graph.add_conditional_edges("accountant", self._select_agent)
        graph.add_edge("calculator", "accountant")
        graph.add_edge("deductions", "accountant")
        graph.add_edge("percentage", "accountant")
        graph.add_edge("pricer", END)
        graph.set_entry_point("accountant")
        self.graph = graph.compile()

    def process_accountant(self, state: AgentState):
        """
        Determines which agents are required to answer the user's question and generates the prompt for them.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the required agents, their prompt, and chat history.
        """
        if 'reasoning' in state:
            reasoning = state['reasoning']
        else:
            reasoning = []
        if 'chat_history' in state:
            chat_history = state['chat_history']
        else:
            chat_history = []
        if 'token_usage_history' in state:
            token_usage_history = state['token_usage_history']
        else:
            token_usage_history = []
        accountant_answer, token_usage = self.accountant.answer(state['question'], reasoning)
        token_usage_history.append(token_usage)
        print(accountant_answer)
        reasoning.append({"role": "assistant", "content": str(accountant_answer)})
        chat_history.append({"role": "accountant", "content": str(accountant_answer)})
        agent_prompt = None
        final_answer = None
        if 'deductions' in accountant_answer:
            next_agent = "deductions"
            agent_prompt = accountant_answer['deductions']
        elif 'calculator' in accountant_answer:
            next_agent = "calculator"
            agent_prompt = accountant_answer['calculator']
        elif 'percentage' in accountant_answer:
            next_agent = "percentage"
            agent_prompt = accountant_answer['percentage']
        elif 'answer' in accountant_answer:
            next_agent = "pricer"
            final_answer = accountant_answer['answer']
        else:
            next_agent = "pricer"
            final_answer = accountant_answer
        return {'next_agent': next_agent, 'agent_prompt': agent_prompt, 'answer': final_answer,
                'reasoning': reasoning, 'chat_history': chat_history, 'token_usage_history': token_usage_history}

    def _select_agent(self, state: AgentState):
        """
        Return the next agent.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            str: The name of the next agent.
        """
        return state['next_agent']

    def process_calculator(self, state: AgentState):
        reasoning = state['reasoning']
        chat_history = state['chat_history']
        token_usage_history = state['token_usage_history']
        calculator_answer, token_usage = self.calculator.answer(state['agent_prompt'])
        token_usage_history.append(token_usage)
        print(calculator_answer)
        reasoning.append({"role": "user", "content": str(calculator_answer['calculator'])})
        chat_history.append({"role": "calculator", "content": str(calculator_answer)})
        return {"reasoning": reasoning, "chat_history": chat_history, 'token_usage_history': token_usage_history}

    def process_percentage(self, state: AgentState):
        reasoning = state['reasoning']
        chat_history = state['chat_history']
        token_usage_history = state['token_usage_history']
        percentage_answer, token_usage = self.percentage.answer(state['agent_prompt'])
        token_usage_history.append(token_usage)
        print(percentage_answer)
        reasoning.append({"role": "user", "content": str(percentage_answer['percentage'])})
        chat_history.append({"role": "percentage", "content": str(percentage_answer)})
        return {"reasoning": reasoning, "chat_history": chat_history, 'token_usage_history': token_usage_history}

    def process_deductions(self, state: AgentState):
        reasoning = state['reasoning']
        chat_history = state['chat_history']
        token_usage_history = state['token_usage_history']
        deductions_answer, token_usage = self.deductions.answer(state['agent_prompt'])
        token_usage_history.append(token_usage)
        print(deductions_answer)
        reasoning.append({"role": "user", "content": str(deductions_answer['deductions'])})
        chat_history.append({"role": "deductions", "content": str(deductions_answer)})
        return {"reasoning": reasoning, "chat_history": chat_history, 'token_usage_history': token_usage_history}

    def pricer(self, state: AgentState):
        token_usage_history = state['token_usage_history']
        total_tokens = sum([sum(i) for i in token_usage_history])
        input_tokens = token_usage_history[0][0]
        output_tokens = token_usage_history[-1][1]
        reasoning_tokens = total_tokens - input_tokens - output_tokens
        return {"input_tokens": input_tokens, "reasoning_tokens": reasoning_tokens, 'output_tokens': output_tokens}
