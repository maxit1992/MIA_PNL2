# TP2

## Description

In this TP, students are required to create an agents system that can answer questions about multiple candidates based on their CV.

## Requirements

- Python 3.8 or higher.
- Pinecone API Key for contextual search.
- Groq API Key for chat completion.
- langgraph for the agents orchestration.
- Streamlit for the graphical user interface.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/maxit1992/MIA_RL1.git
    cd MIA_PNL2/tp2
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

- Upload up to three CVs in PDF format.
- Automatically process and store CV data in a vector database for contextual search.
- Ask questions about the uploaded CVs, and the system will provide accurate and concise answers.
- Uses agents orchestration to handle the question-answering process.

## Code Quality

No vulnerabilities or code smells were detected by SonarQube analysis.