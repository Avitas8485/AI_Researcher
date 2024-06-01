from typing import List


       
def generate_search_queries_prompt(question: str, num_queries: int = 3) -> str:
    """ Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return f'Write {num_queries} separate google search queries to search online that would provide a comprehensive understanding of "{question}" Write your answer as a json array.'


def check_relevance_prompt(question: str, text: str) -> str:
    """Generates the check relevance prompt for the given question and text.
    Args:
        question (str): The question to generate the check relevance prompt for
        text (str): The text to generate the check relevance prompt for
    Returns:
        str: The check relevance prompt for the given question and text
    """
    return f'You are provided with a query and information extracted from a database. Your task is to simply do a comparison and determine if the information provided provides a relevant and comprehensive answer to the query. Give me your rationale for your decision. \n\n Query: \n{question}\n\n Database Information: \n{text}'

def answer_question_prompt(question: str, db_results: str)-> str:
    """Generates the answer question prompt for the given question and db_results.
    Args:
        question (str): The question to generate the answer question prompt for
        db_results (str): The db_results to generate the answer question prompt for
    Returns:
        str: The answer question prompt for the given question and db_results
    """

    return f'Answer the following question based on the information provided. Be as detailed as possible but do not add anything that is not included: \n\n Query: {question}\n{db_results}'


def research_report_prompt(queries: List[str], research_info: str)-> str:
    """Generates the research report prompt for the given queries and research_info.
    Args:
        queries (str): The queries to generate the research report prompt for
        research_info (str): The research_info to generate the research report prompt for
        Returns:
        str: The research report prompt for the given queries and research_info
        """
    return f"Your task is to provide a comprehensive research report based on the following queries and information. Be as detailed as possible and include any factual information such as numbers, stats, quotes, etc.\n\n Queries\n {queries}\n\nInformation\n{research_info}"

def summarize_text_prompt(question: str, text: str) -> str:
    """Generates the prompt to summarize the text with respect to the question.
    Args:
        question (str): The question to summarize the text with respect to
        text (str): The text to summarize
    Returns:
        str: The prompt to summarize the text with respect to the question
    """
    return f"""Provide a comprehensive explanation of the following query using the information provided. If the question cannot be answered using the text, simply summarize the text itself. Do not include any information that is not in the text and be as detailed as possible. Include any factual information such as numbers, stats, quoter, etc if available\n\n Query: {question}\n\nInformation\n{text}"""   

def final_summary_prompt(question: str, text: str) -> str:
    """Generates the prompt to summarize the text with respect to the question.
    Args:
        question (str): The question to summarize the text with respect to
        text (str): The text to summarize
    Returns:
        str: The prompt to summarize the text with respect to the question
    """
    return f"""Your task is to parse and combine multiple summaries on the same topic to one document. The provided text consists of various sections, each potentially containing a summary relevant to the given query. Unless explicitly stated as irrelevant, all sections should be included in the final output without any modifications or removals.

When presented with a query and the corresponding summaries, follow these steps:

1. Carefully read and understand the query to identify the topic and key information being requested.

2. Analyze each section of the provided text:
   - If a section is marked as "irrelevant," exclude it from the final output.
   - If a section is not marked as irrelevant, consider it as a relevant summary and include it in the final output.


3. Do not modify, remove, or add any content to the individual summaries unless explicitly stated as "irrelevant."


Query: {question}

Information:
{text}""" 

def create_summary_prompt(chunk, question):
    return f""" Summarize the following text based on the task or question: {question}
    If the question cannot be answered using the text, simply summarize the text itself.
    Do not include any information that is not in the text.
    {chunk}
"""

def generate_report_prompt(documentation: str) -> str:

    return f"""You are an advanced AI research assistant, specialized in creating comprehensive research reports based on given queries and information extracted from the internet.

Your task is to analyze the provided queries and accompanying information, and generate a well-structured research report that addresses all aspects of the queries.

When presented with queries and relevant information, follow these steps:

1. Carefully read and understand each query, identifying the key research areas and objectives.

2. Thoroughly analyze the provided information, extracting and organizing the relevant data, facts, and insights related to each query.

3. Structure the research report in a logical and easy-to-follow format, with clear sections and subheadings corresponding to each query or research area.

4. Within each section, present the relevant information in a concise yet comprehensive manner, ensuring that all aspects of the query are addressed.

5. Use proper academic or professional writing style, with clear and well-structured paragraphs, appropriate citations (if applicable), and a consistent tone throughout the report.

6. Include any relevant references, sources, or citations to support the information presented in the report.

7. Provide a well-rounded conclusion that summarizes the key findings and insights from the research.

8. Ensure that the final research report adheres to the expected output format specified in the following documentation:

{documentation}

Your goal is to create a high-quality, comprehensive research report that effectively synthesizes the provided information to address the given queries, serving as a valuable resource for the intended audience.

"""