from SingletonGroq import SingletonGroq
from TextProvider import TextProvider
from VectorDB import VectorDB


class AgentCV:
    """
    This class handles a single agent that answers questions based on a single CV.
    """
    AGENT_CV_PROMPT = """Instructions:
    - You are a helpful agent assistant in an agent system that analyzes chunks of texts extracted from a single candidate's CVs and returns questions' answers about the candidate to a principal agent.
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response."""

    def __init__(self, agent_name: str, cv_file: str):
        """
        Initializes the AgentCV class by setting up the Groq client and saving the CV file to the vector database.

        Args:
            agent_name (str): the candidate's name.
            cv_file (str): the path to the candidate's CV file.
        """
        self.agent_name = agent_name
        self.vector_db = VectorDB(index_name=agent_name.lower().replace(' ', '-'))
        self.client = SingletonGroq().groq
        self._save_cv(cv_file)

    def _save_cv(self, cv_file: str):
        """
        Saves the CV file to the vector database.

        Args:
            cv_file (str): The path to the CV file to be saved.
        """
        text_provider = TextProvider(cv_file)
        text = text_provider.get_chunks(chunk_max_size=512)
        self.vector_db.save_text(text)

    def greetings(self):
        """
        Returns a greeting message from the agent.

        Returns:
            str: A greeting message from the agent.
        """
        return f"Hello, I am a CV agent. I can answer questions about the {self.agent_name}'s CV."

    def answer(self, question: str):
        """
        Generates an answer to a question based on the provided CV context.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The generated answer to the question.
        """
        context = self.vector_db.get_similar_text(question, top_k=3)
        clean_context = '\n'.join(item['metadata']['text'] for item in context['matches'])
        sys_prompt = f"""{self.AGENT_CV_PROMPT}
                
        Context: 
        {clean_context}"""
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
        return chat_completion.choices[0].message.content
