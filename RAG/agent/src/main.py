from dotenv import find_dotenv, load_dotenv
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
from load_index import canada_engine
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# %%
filepath = Path.joinpath(Path.cwd().parent, 'agent_data', 'WorldPopulation2023.csv')
population = pd.read_csv(filepath)
# %%
population_query_engine = PandasQueryEngine(
    df=population, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
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