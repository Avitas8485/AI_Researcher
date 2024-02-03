from typing import List

def generate_search_queries_prompt(question: str) -> str:
    """ Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return f'Write 3 separate google search queries to search online that would provide a comprehensive understanding of "{question}".'
          

def check_relevance_prompt(question: str, text: str) -> str:
    """Generates the check relevance prompt for the given question and text.
    Args:
        question (str): The question to generate the check relevance prompt for
        text (str): The text to generate the check relevance prompt for
    Returns:
        str: The check relevance prompt for the given question and text
    """
    # you are provided with a query and information extracted from a database. Your task is to determine if the information is relevant to the query. Do not add anything that is not provided: \n\n Query: \n{question}\n\n Database Information: \n{text}

    return f'You are provided with a query and information extracted from a database. Your task is to determine if the information is relevant to the query. Do not add anything that is not provided: \n\n Query: \n{question}\n\n Database Information: \n{text}'

def answer_question_prompt(question: str, db_results: str)-> str:
    """Generates the answer question prompt for the given question and db_results.
    Args:
        question (str): The question to generate the answer question prompt for
        db_results (str): The db_results to generate the answer question prompt for
    Returns:
        str: The answer question prompt for the given question and db_results
    """

    return f'Answer the following question based on the information provided: {question}\n{db_results}'

def research_report_prompt(queries: List[str], research_info: str)-> str:
    """Generates the research report prompt for the given queries and research_info.
    Args:
        queries (str): The queries to generate the research report prompt for
        research_info (str): The research_info to generate the research report prompt for
        Returns:
        str: The research report prompt for the given queries and research_info
        """
    return f"Your task is to provide a comprehensive explanation of the following queries based on the information provided. Do not add anything that is not included in the information provided, if necessary, list the sources of the information.\n\n Queries\n {queries}\n\nInformation\n{research_info}"
