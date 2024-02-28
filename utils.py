from unstructured.partition.pdf import partition_pdf
import base64
import numpy as np
from unstructured.documents.elements import Element
import pandas as pd
from io import StringIO
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def pretty_print_element(element: Element):
    print("-------- Element --------")
    print(f"Element type: {type(element)}")
    print(f"Element text: {element.text}")
    if element.category == "Table":
        print(f"HTML table: {element.metadata.text_as_html}")


def html_table_df(html_table: str) -> pd.DataFrame:
    return pd.read_html(StringIO(html_table), encoding='utf-8')[0]


def df_to_table_str(df: pd.DataFrame) -> str:
    str_columns = " | ".join(df.columns)
    table_str = str_columns + "\n" + "-"*len(str_columns) + "\n"
    values = df.values.tolist()
    for row in values:
        table_str += " | ".join(str(el) for el in row) + "\n"
    return table_str


def html_table_to_pipe_table(html_table: str) -> str:
    df = html_table_df(html_table)
    return df_to_table_str(df)


def ask_question(llm, prompt_text: str, extracted_table: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    summarize_chain = (
        {
            "question": itemgetter("question"),
            "extracted_table": itemgetter("extracted_table")
        }
        |
        prompt
        |
        llm
        |
        StrOutputParser()
    )
    answer = ""
    for chunk in summarize_chain.stream(input={"question": question, "extracted_table": extracted_table}):
        answer += chunk

    return answer


def cosine_similarity(v1, v2) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_pdf_elements(pdf_path: str, detected_image_directory: str):
    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        include_page_breaks=True,
        chunking_strategy='auto',
        extract_image_block_output_dir=detected_image_directory
    )
    return elements
