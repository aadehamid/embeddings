# import weaviate
# import weaviate.classes as wvc
# import os
# client.connect()
# properties = [
#                 wvc.config.Property(
#                 name="OriginalShorttext",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Description of the issue."
#             ),
#
#                 wvc.config.Property(
#                 name="MajorSystem",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Name of the system."
#             ),
#                wvc.config.Property(
#                 name="Part",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Part of the system"
#             ),
#
#                wvc.config.Property(
#                 name="Action",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Action taken to resolve the issue."
#                ),
#                 wvc.config.Property(
#                 name="FM",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Symptoms that the issue is related to."
#                 ),
#                 wvc.config.Property(
#                 name="Location",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Location of the issue on the system"
#                 ),
#                 wvc.config.Property(
#                 name="FuncLocation",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Functional location of the issue."
#                 ),
#                  wvc.config.Property(
#                 name="Asset",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Asset that the work order is related to."
#                 ),
#                 wvc.config.Property(
#                 name="Cost",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Cost of the work order."
#                 ),
#                 wvc.config.Property(
#                 name="RunningTime",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Running time of the work order."
#                 ),
#
#                 wvc.config.Property(
#                 name="Variant",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Variant of the work order."
#                 ),
#                 wvc.config.Property(
#                 name="Comments",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Comments on the work order."
#                 ),
#                 wvc.config.Property(
#                 name="SuspSugg",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Suggested suspension."
#                 ),
#                 wvc.config.Property(
#                 name="Rule",
#                 data_type=wvc.config.DataType.TEXT,
#                 vectorize_property_name=True,  # Skip the property name ("OriginalShorttext") when vectorizing
#                 tokenization=wvc.config.Tokenization.LOWERCASE,  # Use "lowecase" tokenization
#                 description="Rule for the work order."
#                 ),
#             ]
# try:
#     work_order_schema = client.collections.create(
#         name="WorkOrder",
#         vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),    # Set the vectorizer to "text2vec-openai" to use the OpenAI API for vector-related operations
#         generative_config=wvc.config.Configure.Generative.cohere(),             # Set the generative module to "generative-cohere" to use the Cohere API for RAG
#         properties= properties)
#
# finally:
#     client.close()