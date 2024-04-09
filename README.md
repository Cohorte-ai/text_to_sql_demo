# Text-to-SQL: Extracting Insights from Databases using OpenAI GPT-3.5 and LlamaIndex
Welcome to the Text-to-SQL demo project using Llama-Index! This application is an updated version of the NA2SQL app, designed to demonstrate the innovative use of Large Language Models (LLMs) in translating natural language queries into SQL queries, and fetching meaningful insights from a database. The database has been expanded to include more records, and the UI now features Streamlit pills, allowing users to easily run sample queries.

## Demo Features
- Natural Language Understanding: Converts user-inputted natural language queries into accurate SQL queries.
- Data Retrieval and Analysis: Retrieves results from the database and interprets them to provide meaningful insights.
- Accessibility: Designed for users with no SQL background to easily interact with and extract insights from databases.

## Potential Enhancements
- Expansion to include more diverse data sources and databases.
- Advanced natural language processing for more intricate query handling.

Try the demo here!

## Tools and Technologies Used
- LLM: OpenAI's GPT-3.5
- LLM Orchestration: Llama-Index==0.10.26
- Data Management: SQLDatabase with SQLite
- UI Framework: Streamlit

## Project Structure
- `app.py`: The main application script for the Streamlit app.
- `Dockerfile`: Contains the Docker configuration for building and running the app.
- `requirements.txt`: Lists the Python packages required for this project.
- `.env`: File to include `OPENAI_API_KEY` for authentication.

## Setup and Usage

### Clone the Repository
```
git clone https://github.com/Cohorte-ai/text_to_sql_demo/
```

### Install Required Packages

```
pip install -r requirements.txt
```

### Run the Streamlit App
```
streamlit run app.py
```

### Docker Support
To build and run the application using Docker, follow these steps:

#### Build the Docker Image
```
docker build -t text-to-sql-app .
```
#### Run the Docker Container
```
docker run -p 8501:8501 --env-file .env text-to-sql-app
```
Note: Ensure you have the `.env` file containing `OPENAI_API_KEY` in your project directory before running the Docker container.

