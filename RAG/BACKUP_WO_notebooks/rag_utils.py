import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
# from trulens_eval import Feedback, TruLlama, OpenAI

# from trulens_eval.feedback import Groundedness
import nest_asyncio
# import to set up weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

nest_asyncio.apply()

import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ----------------------------------------------
# Load data to sqlite DB
# ----------------------------------------------

import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data_to_sql_db(filepath: str, dbpath: str, tablename: str) -> \
        (sqlite3.Connection, create_engine, pd.DataFrame):
    """
    Load data from a CSV file into a SQLite database.

    Args:
    - filepath (str): The path to the CSV file.
    - dbpath (str): The path to the SQLite database.
    - tablename (str): The name of the table to create in the database.


    Returns:
    - conn (sqlite3.Connection): The connection to the SQLite database.
    - data (pd.DataFrame): The data from the CSV file as a DataFrame.
    - engine (create_engine): The SQLite engine.
    """
    # Read the data from the CSV file into a pandas DataFrame
    data = pd.read_csv(filepath)

    # Drop the 'Unnamed: 16' column from the DataFrame
    data.drop("Unnamed: 16", axis=1, inplace=True)

    # Create a connection to the SQLite database
    conn = sqlite3.connect(dbpath)
    engine = create_engine("sqlite:///" + dbpath)

    # Write the DataFrame to a table in the SQLite database
    data.to_sql(tablename, conn, if_exists='replace', index=False)

    # Return the connection to the SQLite database
    return conn, engine, data


# ----------------------------------------------
# Create text to query engine
from typing import Union
from llama_index.core import set_global_service_context
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine

from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema


def text_to_query_engine(model_name: str, embedding_model_name: str, table_name: str, engine: str,
                         temperature: float = 0.1, not_know_table: bool = True) -> Union[
    NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine]:
    """
    Convert text to a query engine for a SQL database.

    This function initializes the necessary components for querying a SQL database using OpenAI models. It sets up both
    the language model and the embedding model from OpenAI, configures the service context, and initializes the SQL database
    connection. Depending on whether the table to be queried is known ahead of time, it returns either a NLSQLTableQueryEngine
    or a SQLTableRetrieverQueryEngine.

    Args:
        model_name (str): The name of the OpenAI model to use.
        embedding_model_name (str): The name of the OpenAI embedding model to use.
        table_name (str): The name of the table in the SQL database to query.
        engine (str): The engine to use for the SQL database.
        temperature (float, optional): The temperature to use for the OpenAI model. Defaults to 0.1.
        not_know_table (bool, optional): Whether the table to query is not known ahead of time. Defaults to True.

    Returns:
        Tuple[Union[NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine],
        SQLDatabase, ServiceContext]: The query engine for the SQL database.
    """
    # Initialize the OpenAI model with the specified temperature and model name
    llm = OpenAI(temperature=temperature, model=model_name)

    # Initialize the OpenAI embedding model with the specified model name
    embed_model = OpenAIEmbedding(model=embedding_model_name)

    # Create a service context with the initialized models
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    # Set the global service context for further use in the application
    set_global_service_context(service_context)

    # Initialize the SQL database with the specified engine and include the table to be queried
    sql_database = SQLDatabase(engine, include_tables=[table_name])

    if not_know_table:
        # If the table to query is not known ahead of time, use SQLTableRetrieverQueryEngine
        # This involves creating a mapping and schema objects for the SQL tables
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [SQLTableSchema(table_name=table_name)]  # Create a schema object for the table
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        # Initialize the query engine with the SQL database and the object index for retrieving tables
        query_engine = SQLTableRetrieverQueryEngine(
            sql_database, obj_index.as_retriever(similarity_top_k=1)
        )
    else:
        # If the table to query is known ahead of time, use NLSQLTableQueryEngine
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=[table_name])

    # Return the initialized query engine
    return query_engine, sql_database, service_context


# ----------------------------------------------
# Create query router engine
from llama_index.core.tools import QueryEngineTool
from typing import List
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine import NLSQLTableQueryEngine


def setup_query_engines(sql_database, doc_index, table_name) -> RouterQueryEngine:
    """
    Sets up SQL and vector query engines and combines them into a router query engine.

    Args:
        sql_database: The SQL database connection or configuration.
        product_index: The index used for sentence window query engine.

    Returns:
        RouterQueryEngine: A router query engine configured with SQL and vector query engines.
    """
    # Set up text2SQL prompt
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
    )

    vector_query_engine = get_sentence_window_query_engine(
        sentence_index=doc_index,
        similarity_top_k=6,
        rerank_top_n=2,
    )

    # Initialize QueryEngineTool for both SQL and vector query engines
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over a work order listing table"
        ),
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for answering semantic questions about historical work order.",
    )

    # Combine the query engines into a router query engine
    query_router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[sql_tool, vector_tool],
    )

    return query_router_engine


# ----------------------------------------------
# Get response

from typing import Any, Union
from llama_index.core.query_engine import RetrySourceQueryEngine
from llama_index.core.query_engine import RetryGuidelineQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator, GuidelineEvaluator
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core import Response


def get_retry_guideline_response(query_router_engine: Union[RetryGuidelineQueryEngine, RetrySourceQueryEngine],
                                 query_str: str, guideline=True) -> Response:
    """
    Sets up query engines with retry mechanisms based on relevancy and guidelines,
    and executes a test query.

    Args:
        query_router_engine (QueryEngine): The main query engine that routes queries.
        query_str (str): The query string to get answer using the retry mechanisms.

    Returns:
        Response: The response from the retry guideline query engine.
    """

    if guideline:

        # Initialize the guideline evaluator with default and additional guidelines
        guideline_eval = GuidelineEvaluator(
            guidelines=DEFAULT_GUIDELINES
                       + "\nThe response should not be overly long.\n"
                         "The response should try to summarize where possible.\n"
                         "First, answer the question\n"
                         "Second provide the reason, why you choose that answer.\n"
        )

        # Set up the retry guideline query engine with the guideline evaluator
        retry_query_router_engine = RetryGuidelineQueryEngine(
            retry_source_query_engine, guideline_eval, resynthesize_query=True,
            max_retries=2,
        )
    else:
        # Initialize the relevancy evaluator
        query_response_evaluator = RelevancyEvaluator()

        # Set up the retry source query engine with the relevancy evaluator
        retry_query_router_engine = RetrySourceQueryEngine(
            query_router_engine, query_response_evaluator
        )

    # Execute the test query using the retry guideline query engine
    retry_response = retry_query_router_engine.query(query_str)

    # Return the response from the retry guideline query engine
    return retry_response


# ----------------------------------------------
# Evaluate if the response adheres to the guidelines
from typing import Tuple, Any
from llama_index.core import Response
from llama_index.core.indices.query.query_transform.feedback_transform import FeedbackQueryTransformation
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.evaluation import GuidelineEvaluator


def evaluate_and_transform_query(
        query_str: str,
        retry_response: Response,
        # Type hint as `Any` since it could be a `Response` or another type with a `.get_response()` method.
        DEFAULT_GUIDELINES: str
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
        guidelines=DEFAULT_GUIDELINES
                   + "\nThe response should not be overly long.\n"
                     "The response should try to summarize where possible.\n"
                     "First, answer the question\n"
                     "Second provide the reason, why you choose that answer.\n"
    )

    # Get the typed response based on the type of `retry_guideline_response`.
    typed_response = (
        retry_response if isinstance(retry_response, Response) else retry_response.get_response()
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


# ----------------------------------------------

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


# openai = OpenAI()
#
# qa_relevance = Feedback(
#     openai.relevance_with_cot_reasons, name="Answer Relevance"
# ).on_input_output()
#
# qs_relevance = (
#     Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
#     .on_input()
#     .on(TruLlama.select_source_nodes().node.text)
#     .aggregate(np.mean)
# )
#
# # grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
# grounded = Groundedness(groundedness_provider=openai)
#
# groundedness = (
#     Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
#     .on(TruLlama.select_source_nodes().node.text)
#     .on_output()
#     .aggregate(grounded.grounded_statements_aggregator)
# )
#
# feedbacks = [qa_relevance, qs_relevance, groundedness]
#
#
# def get_trulens_recorder(query_engine, feedbacks, app_id):
#     tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
#     return tru_recorder
#
#
# def get_prebuilt_trulens_recorder(query_engine, app_id):
#     tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
#     return tru_recorder


from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
import os


def build_sentence_window_index(
        document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def build_sentence_window_index_vector_DB(
        document, client, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index", prefix=None):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    sentence_index = VectorStoreIndex.from_documents([document], storage_context=storage_context,
                                                     service_context=sentence_context)

    return sentence_index


def get_sentence_window_query_engine(
        sentence_index,
        similarity_top_k=6,
        rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index",
        chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def build_automerging_index_vector_DB(
        documents,
        client,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        chunk_sizes=None,
        prefix=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    auto_merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix=prefix)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)
    automerging_index = VectorStoreIndex.from_documents(leaf_nodes, storage_context=storage_context,
                                                        service_context=auto_merging_context)

    return automerging_index


def get_automerging_query_engine(
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
