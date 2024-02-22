import sqlite3

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine

load_dotenv(find_dotenv())
import os
import openai
openai.api_key = os.getenv("OPENA_AI_KEY")
from pathlib import Path
import pandas as pd

from llama_index.core.query_engine import PandasQueryEngine
# from prompt import new_prompt, instruction_str, context

from prompt import (
    context, instruction_str,
    new_prompt)
from note_engine import note_engine
from load_index import canada_engine, text_to_query_engine
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine


# %%
filepath = Path.joinpath(Path.cwd().parent, 'agent_data', 'WorldPopulation2023.csv')
population = pd.read_csv(filepath)
dbpath = str(Path(os.path.join(os.path.dirname(os.getcwd()), 'agent_data',
                'WorldPopulation2023.db')))
# Connect to sqlite db
conn = sqlite3.connect(dbpath)
engine = create_engine("sqlite:///" + dbpath)
# %%
# population_query_engine = PandasQueryEngine(
#     df=population, verbose=True, instruction_str=instruction_str
# )
# population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-small"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)
Settings.llm = llm
Settings.embed_model = embed_model
table_name = "population"

_, sql_database = text_to_query_engine(model_name, embedding_model_name, table_name, engine)

# Set up text2SQL prompt
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[table_name],)
# %%
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=sql_query_engine,
        metadata=ToolMetadata(
            name="sql_query_engine",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)