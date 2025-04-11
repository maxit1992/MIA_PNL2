import os

from groq import Groq


class Chat:

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def answer(self, question: str, context: {}):
        # Assuming the context is a list of dictionaries with 'text' key
        clean_context = '\n'.join("{'" + item['metadata']['text'] + "'}" for item in context['matches'])
        sys_prompt = f"""Instructions:
- You are a helpful assistant that analyzes chunks of texts extracted from a candidate's CVs and answers questions about the candidate.
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
        
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
