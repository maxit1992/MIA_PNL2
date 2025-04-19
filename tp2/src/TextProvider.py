import ast

import pymupdf

from SingletonGroq import SingletonGroq


class TextProvider:
    """
    This class provides texts from files.
    """

    def __init__(self, file: str):
        """
        Initializes the TextProvider with the given file.

        Args:
            file (str): The path to the file to be processed.
        """
        self.file = file
        self.client = SingletonGroq().groq

    def get_text(self) -> str:
        """
        Extracts clean text from the file.

        Returns:
            str: The cleaned text extracted from the file.
        """
        text = ""
        for page in pymupdf.open(self.file):
            text += page.get_text()
        text = text.replace("\n", " ")
        # Remove extra spaces
        text = " ".join(text.split())
        return text

    def get_chunks(self, chunk_max_size: int) -> list[str]:
        """
        Splits the extracted text into smaller chunks of a specified maximum size using an LLM to provide sentences with
        full meaning and context.

        Args:
            chunk_max_size (int): The maximum size of each text chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        text = self.get_text()
        chunks = []
        topic = ''
        while len(text) > 0:
            chunk_candidate = text[:chunk_max_size]
            sys_prompt = f"""Instructions:
            - You are a helpful assistant that receives chunks from a candidate's CV (extracted using a fixed number of words) and returns a subset of the chunk with full meaning and context.
            - Do not modify the words int the input chunk.
            - Use the topic of the previous chunk to determine the scope of the new chunk. - Previous topic: '{topic}'
            - Use the following format to return the text: {{'chunk': 'the chunk', 'topic': 'a topic for the chunk different from the previous one'}}"""
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": chunk_candidate,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            try:
                result = ast.literal_eval(chat_completion.choices[0].message.content)
                if 'topic' in result and 'chunk' in result:
                    topic = result.get('topic')
                    content = result.get('chunk')
                    chunks.append(f"{{'topic': '{topic}', 'content': '{content}}}")
                    # Approximate char count with answer len
                    text = text[len(content):]
                else:
                    raise ValueError("Invalid response format")
            except (Exception,):
                # If LLM failed, we use the chunk as is
                chunks.append(f"{{'topic': 'unknown', 'chunk': {chunk_candidate}}}")
                text = text[chunk_max_size:]
        return chunks
