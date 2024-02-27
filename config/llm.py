import os
from langchain_openai import AzureChatOpenAI

DEPLOYMENT_NAME_GPT_3_5 = 'gpt35'
DEPLOYMENT_NAME_GPT_4 = 'gpt4'
DEPLOYMENT_NAME_GPT4_VISION = 'gpt4-vision-switzerland'  # gpt4 vision preview only available in SWITZERLAND

GPT_35 = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT_4"),
    openai_api_version='2023-05-15',
    deployment_name=DEPLOYMENT_NAME_GPT_3_5,
    openai_api_key=os.getenv("OPENAI_API_KEY_GPT_4"),
    openai_api_type="azure",
    temperature=0,
)

GPT_4 = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT_4"),
    openai_api_version='2023-05-15',
    deployment_name=DEPLOYMENT_NAME_GPT_4,
    openai_api_key=os.getenv("OPENAI_API_KEY_GPT_4"),
    openai_api_type="azure",
    temperature=0,
)

GPT_4_V = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT4_VISION"),
    openai_api_version="2023-07-01-preview",
    deployment_name=DEPLOYMENT_NAME_GPT4_VISION,
    openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_VISION"),
    openai_api_type="azure",
    temperature=0.7,
    max_tokens=4000,
    kawargs={"top_p": 0.95},
)
