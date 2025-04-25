# TP3

## Description

In this TP, students are required to create an agents system that emulates reasoning to solve a complex problem.
In this implementation, a tax bot is created to answer how much tax a person has to pay. The system takes into account a
tax table and applicable deductions.
The system has 4 agents: accountant, calculator, deductions manager and tax table manager.

## Requirements

- Python 3.8 or higher.
- Groq API Key for chat completion.
- langgraph for the agents orchestration.
- Streamlit for the graphical user interface.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/maxit1992/MIA_PNL2.git
    cd MIA_PNL2/tp3
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To start the tax bot application, run the following command:

```sh
streamlit run src/main.py
```

This will launch the Streamlit app in your default web browser. Then you have to ask the bot how much tax you have to
pay, providing your monthly income, your expenses that may be deductible and the current month.

## Features

- LLM with chain of thoughts to reason about the tax calculation step by step.
- Tax table configurable with a csv file.
- Deductions configurable with a csv file.

## Code Quality

No vulnerabilities or code smells were detected by SonarQube analysis.