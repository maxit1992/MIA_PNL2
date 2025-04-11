import pymupdf


class TextProvider:

    def __init__(self, file: str):
        self.file = file

    def get_text(self) -> str:
        text = ""
        for page in pymupdf.open(self.file):
            text += page.get_text()
        text = text.replace("\n", " ")
        # Remove extra spaces
        text = " ".join(text.split())
        return text

    def get_chunks(self, chunk_max_size: int) -> list[str]:
        text = self.get_text()
        chunks = []
        while len(text) > chunk_max_size:
            chunk = text[:chunk_max_size]
            text = text[chunk_max_size:]
            chunks.append(chunk)
        if text:
            chunks.append(text)
        return chunks
