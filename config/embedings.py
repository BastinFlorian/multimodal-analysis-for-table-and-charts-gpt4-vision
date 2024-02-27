import os
from langchain_community.embeddings import AzureOpenAIEmbeddings

AZURE_ADA_002_EMBEDDINGS = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT_4"),
    api_key=os.getenv("OPENAI_API_KEY_GPT_4"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
)
