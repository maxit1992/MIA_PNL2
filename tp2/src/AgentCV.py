import os

from groq import Groq

from TextProvider import TextProvider
from VectorDB import VectorDB


class AgentCV:
    """
    This class handles a single agent that answers questions based on a single CV.
    """
    AGENT_CV_PROMPT = """Instructions:
    - You are a helpful agent assistant in an agent system that analyzes chunks of texts extracted from a single candidate's CVs and returns questions answer about the candidate to a principal agent.
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response."""

    def __init__(self, agent_name: str, cv_file: str):
        """
        Initializes the Chat class by setting up the Groq client with the API key.
        """
        self.agent_name = agent_name
        self.vector_db = VectorDB(index_name=agent_name.lower().replace(' ', '-'))
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        #self.save_cv(cv_file)

    def save_cv(self, cv_file: str):
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
        Generates an answer to a user's question based on the provided context.

        Args:
            question (str): The user's question to be answered.
            context (dict): A dictionary containing context information, typically a list of matches with metadata.

        Returns:
            str: The generated answer to the question.
        """
        # Assuming the context is a list of dictionaries with 'text' key
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
        return f"{{'agent':'{self.agent_name}', 'content':'{chat_completion.choices[0].message.content}'}}"
