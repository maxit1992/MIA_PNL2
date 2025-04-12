# TP1

## Description

In this TP, students are required to create a chatbot that can answer questions about a candidate based on their CV.
The bot must use RAG for specific information retrieval.

## Requirements

- Python 3.8 or higher.
- Pinecone API Key for contextual search.
- Groq API Key for chat completion.
- Streamlit for the graphical user interface.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/maxit1992/MIA_RL1.git
    cd MIA_PNL2/tp1
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To start the CV Chat Bot application, run the following command:

```sh
streamlit run src/main.py
```

This will launch the Streamlit app in your default web browser.

## Features

- Upload CVs: users can upload CVs in PDF format. CVs are converted to text, split into chunks, and stored in a vector
  database to enable RAG.
- Chatbot Interaction: users can ask questions about the uploaded CVs, and the bot provides answers based on the
  extracted information.

## Code Quality

No vulnerabilities or code smells were detected by SonarQube analysis.