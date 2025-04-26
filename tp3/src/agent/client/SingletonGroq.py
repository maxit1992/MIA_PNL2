import os

from groq import Groq


class SingletonGroq:
    """
    This class implements a singleton pattern to ensure that only one instance of the Groq client is created and shared
    throughout the application.
    """
    _instance = None
    groq = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the class is created. If an instance already exists, it returns the existing
        instance.
        """
        if not cls._instance:
            cls._instance = super(SingletonGroq, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the Groq client if it has not already been initialized, using the API key from the GROQ_API_KEY
        environment variable.
        """
        if self.groq is None:
            self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
