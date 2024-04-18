import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.core import Settings

from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.llms.openai import OpenAI
import openai
import os
import pandas as pd
from streamlit_pills import pills
from PIL import Image


# from llama_index.llms.palm import PaLM


from llama_index.core import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import sqlite3


from llama_index.core import SQLDatabase, ServiceContext

from llama_index.core.query_engine import NLSQLTableQueryEngine

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


class StreamlitChatPack(BaseLlamaPack):

    def __init__(
        self,
        page: str = "Natural Language to SQL Query",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""

        self.page = page

    def load_css(self, file_name):
        with open(file_name) as css:
            st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import streamlit as st

        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        # Load your CSS
        self.load_css('style.css')

        # Load your image
        image = Image.open('logo.png')
        # Display the image in the sidebar at the top left
        st.sidebar.image(image, width=40)

        if "messages" not in st.session_state:  # Initialize the chat messages history
            st.session_state["messages"] = [
                {"role": "assistant",
                    "content": f"Hello. Ask me anything related to the database."}
            ]

        st.title(
            f"{self.page}💬"
        )
        st.info(
            f"Explore database views with this AI-powered app. Pose any question and receive exact SQL queries.",
            icon="ℹ️",
        )
        # Define the pills with emojis
        query_options = ["None", "In a markdown table format show which users bought '4K LED Smart TV', their purchase date and their location",
                         "In a markdown table show all the products bought under books category and their reviews", "Analyse all the reviews for Electronics category and list points of improvements in a table"]
        # emojis = ["👥", "📅", "🏷️"]
        selected_query = pills(
            "Select example queries or enter your own query in the chat input below", query_options, key="query_pills")

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(message)  # Add response to message history

        def get_table_data(table_name, conn):
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            return df

        # @st.cache_resource
        def load_db_llm():
            # Load the SQLite database
            # engine = create_engine("sqlite:///ecommerce_platform1.db")
            engine = create_engine(
                "sqlite:///ecommerce_platform1.db?mode=ro", connect_args={"uri": True})

            sql_database = SQLDatabase(engine)  # include all tables

            # Initialize LLM
            # llm2 = PaLM(api_key=os.environ["GOOGLE_API_KEY"])  # Replace with your API key
            Settings.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")

            # Settings.embed_model = "local:BAAI/bge-base-en-v1.5"

            return sql_database, Settings, engine

        sql_database, service_context, engine = load_db_llm()

       # Sidebar for database schema viewer
        st.sidebar.markdown("## Database Schema Viewer")

        # Create an inspector object
        inspector = inspect(engine)

        # Get list of tables in the database
        table_names = inspector.get_table_names()

        # Sidebar selection for tables
        selected_table = st.sidebar.selectbox("Select a Table", table_names)

        db_file = 'ecommerce_platform1.db'
        conn = sqlite3.connect(db_file)

        # Display the selected table
        if selected_table:
            df = get_table_data(selected_table, conn)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)

        # Close the connection
        conn.close()

        st.sidebar.markdown('## Disclaimer')
        st.sidebar.markdown(
            """This application is for demonstration purposes only and may not cover all aspects of real-world data complexities. Please use it as a guide and not as a definitive source for decision-making.""")

        if "query_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["query_engine"] = NLSQLTableQueryEngine(
                sql_database=sql_database,
                synthesize_response=True,
                service_context=service_context
            )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # To avoid duplicated display of answered pill questions each rerun
        if selected_query and selected_query != "None" and selected_query not in st.session_state.get(
            "displayed_pill_questions", set()
        ):
            with st.spinner():
                st.session_state.setdefault("displayed_pill_questions", set()).add(selected_query)
                with st.chat_message("user"):
                    st.write(selected_query)
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query(
                        "User Question:"+selected_query+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    add_to_message_history("user", selected_query)
                    add_to_message_history("assistant", sql_query)

        # Prompt for user input and save to chat history
        prompt = st.chat_input("Enter your natural language query about the database") 
        if prompt:
            with st.spinner():
                add_to_message_history("user", prompt)

                # Display the new question immediately after it is entered
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query(
                        "User Question:"+prompt+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    add_to_message_history("assistant", sql_query)


if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
