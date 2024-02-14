# %%
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import os

# -----------------------------------------------------------------------------
# openai and weaviate
import openai

openai.api_key = os.getenv("OPENA_AI_KEY")
import weaviate

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_APIKEY = os.getenv("WEAVIATE_API_KEY")
from weaviate.auth import AuthApiKey

# -----------------------------------------------------------------------------
# import utilities
from rag_utils import (
    load_data_to_sql_db,
    text_to_query_engine,
    build_sentence_window_index_vector_DB,
    setup_query_engines,
    get_retry_guideline_response,
    evaluate_and_transform_query,
    build_sentence_window_index,
)

# from RAG.WO_notebooks.rag_utils import build_sentence_window_index_vector_DB


# -----------------------------------------------------------------------------
# llama_index imports
# --------------------------------------------------------------------------------
from llama_index.core.response.notebook_utils import display_response
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.weaviate import WeaviateVectorStore


# %%
model_name = "gpt-3.5-turbo"
embedding_model_name = "text-embedding-3-large"
# embedding_model_name="local:BAAI/bge-small-en-v1.5"
llm = OpenAI(temperature=0.1, model=model_name)
embed_model = OpenAIEmbedding(model=embedding_model_name)

# %%
auth_config = AuthApiKey(api_key=WEAVIATE_APIKEY)

client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)

# %%
# # Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=WEAVIATE_URL,
#     auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_APIKEY))

# %% [markdown]
# # Load Work Order Data

# %%
# path to the raw dataset
table_name = "work_order_table"
dbpath = "./data/wo_data.db"
wo_data_path = "/Users/hamidadesokan/Dropbox/2_Skill_Development/DLML/genai_applications/embeddings/RAG/WO_notebooks/data/excavator_2015_cleaned_forpdl.csv"
conn, engine, data = load_data_to_sql_db(wo_data_path, dbpath, table_name)

# %% [markdown]
# # test the query engine

# %%
# model_name = "gpt-3.5-turbo"
# embedding_model_name = "text-embedding-3-large"
# llm = OpenAI(temperature=0.1, model = model_name)
# embed_model = OpenAIEmbedding(model= embedding_model_name)
# query_engine,sql_database, service_context = text_to_query_engine(model_name, embedding_model_name, table_name, engine)
# # query_str = "Which work oder cost 183.05?"
# query_str = "How much in total did we spend in 2011?"
# response = query_engine.query(query_str)
# display_response(response)

# %% [markdown]
# # Create document object

# %%
columns_to_embed = [
    "OriginalShorttext",
    "MajorSystem",
    "Part",
    "Action",
    "FM",
    "Location",
    "FuncLocation",
]

columns_to_metadata = [
    "Asset",
    "Cost",
    "RunningTime",
    "Variant",
    "Comments",
    "SuspSugg",
    "Rule",
]


docs = []
for i, row in data.iterrows():
    to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
    values_to_embed = {k: str(row[k]) for k in columns_to_embed if k in row}
    to_embed = "\n".join(
        f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
    )
    newDoc = Document(text=to_embed, metadata=to_metadata)
    docs.append(newDoc)

# Create a single document from a list of Documents
# this is what we will chunk up and store with its embedding in Weaviate
document = Document(text="\n\n".join([doc.text for doc in docs]))

# %% [markdown]
# # Build Sentence window

# %%
# load work_order_schema from json
client.schema.delete_class("WorkOrder")
import json

with open("./config/work_order_schema", "r") as f:
    work_order_schema = json.load(f)

client.schema.create(work_order_schema)
print("Product schema was created.")


# %%
work_order_index = build_sentence_window_index_vector_DB(
    document=[document],
    client=client,
    llm=llm,
    embed_model=embed_model,
    prefix="Work_order_sent_win_index",
)

# %% [markdown]
# # Save the index in vectore store

# %%
query_router_engine = setup_query_engines(sql_database, work_order_index, table_name)
response = get_retry_guideline_response(
    query_router_engine, "How much in total did we spend in 2011?", guideline=False
)
print(response)

# %%
query_router_engine

# %%
# query_str = "Which work oder cost 183.05?"
query_str = "How much in total did we spend in 2011?"
response = query_router_engine.query(query_str)
print(str(response))
# display_response(response)

# %%
data.loc[data.BscStartDate.str.startswith("2011")].Cost.sum()

# %% [markdown]
#

# %% [markdown]
# # Evaluating query and prompts

# %%
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.evaluation import GuidelineEvaluator

# Evaluating query and prompts
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.evaluation import GuidelineEvaluator
from llama_index.core import Response
from llama_index.core.indices.query.query_transform.feedback_transform import FeedbackQueryTransformation

# Guideline eval
guideline_eval = GuidelineEvaluator(
    guidelines=DEFAULT_GUIDELINES + "\nThe response should not be overly long.\n"
    "The response should try to summarize where possible.\n"
    "First, answer the question\n"
    "Second provide the reason, why you choose that answer.\n"
)  # just for example

typed_response = (
    retry_guideline_response
    if isinstance(retry_guideline_response, Response)
    else retry_guideline_response.get_response()
)
eval = guideline_eval.evaluate_response(query_str, typed_response)
print(f"Guideline eval evaluation result: {eval.feedback}")

feedback_query_transform = FeedbackQueryTransformation(resynthesize_query=True)
transformed_query = feedback_query_transform.run(query_str, {"evaluation": eval})
print(f"Transformed query: {transformed_query.query_str}")

# %%
from typing import Tuple, Any
from llama_index.core import Response
from llama_index.core.indices.query.query_transform.feedback_transform import FeedbackQueryTransformation
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.evaluation import GuidelineEvaluator


def evaluate_and_transform_query(
    query_str: str,
    retry_response: Response,  # Type hint as `Any` since it could be a `Response` or another type with a `.get_response()` method.
    DEFAULT_GUIDELINES: str,
) -> Tuple[GuidelineEvaluator, FeedbackQueryTransformation]:
    """
    Evaluates a query response against a set of guidelines and transforms the query based on the feedback.

    Args:
        query_str (str): The original query string.
        retry_guideline_response (Any): The initial response to evaluate, can be a `Response` object or another type with a `get_response()` method.
        DEFAULT_GUIDELINES (str): The default guidelines string to be used for evaluation.

    Returns:
        Tuple[str, str]: A tuple containing the guideline evaluation feedback and the transformed query string.
    """
    # Initialize the guideline evaluator with additional guidelines.
    guideline_eval = GuidelineEvaluator(
        guidelines=DEFAULT_GUIDELINES + "\nThe response should not be overly long.\n"
        "The response should try to summarize where possible.\n"
        "First, answer the question\n"
        "Second provide the reason, why you choose that answer.\n"
    )

    # Get the typed response based on the type of `retry_guideline_response`.
    typed_response = (
        retry_response
        if isinstance(retry_response, Response)
        else retry_response.get_response()
    )

    # Evaluate the response against the guidelines.
    eval = guideline_eval.evaluate_response(query_str, typed_response)
    print(f"Guideline eval evaluation result: {eval.feedback}")

    # Transform the query based on feedback.
    feedback_query_transform = FeedbackQueryTransformation(resynthesize_query=True)
    transformed_query = feedback_query_transform.run(query_str, {"evaluation": eval})
    print(f"Transformed query: {transformed_query.query_str}")

    # Return the feedback and the transformed query string.
    return eval.feedback, transformed_query.query_str


# %%
feedback, transformed_query = evaluate_and_transform_query(
    query_str, response, DEFAULT_GUIDELINES
)

# %%
print(transformed_query)

# %%
print(feedback)

# %%
