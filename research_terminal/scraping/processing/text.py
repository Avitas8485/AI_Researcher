from typing import Generator, Optional
from selenium.webdriver.remote.webdriver import WebDriver 
from research_terminal.llm.llama_model import LlamaModel
from research_terminal.llm.grammar.pydantic_models import Summary, ArticleSummary
from research_terminal.llm.grammar.pydantic_models_to_grammar import generate_gbnf_grammar_and_documentation
from research_terminal.llm.prompts import summarize_text_prompt, final_summary_prompt
from llama_cpp.llama_grammar import LlamaGrammar
import json
import os
from research_terminal.logger.logger import logger


def split_text(text: str, max_length: int = 8000) -> Generator[str, None, None]:
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 8192.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: If the text is longer than the maximum length
    """
    logger.info(f"Splitting text into chunks of length {max_length}")
    paragraphs = text.split("\n")
    
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1
    logger.info(f"Text split into {len(current_chunk)} chunks")
    if current_chunk:
        yield "\n".join(current_chunk)


def summarize_text(question: str, text: str, max_length: int = 8000, driver: Optional[WebDriver] = None) -> ArticleSummary:
    """Summarizes the text with respect to the question.
    Args:
        question (str): The question to summarize the text with respect to
        text (str): The text to summarize
    Returns:
        Summary: The summary of the text with respect to the question
    """
    summaries = []
    chunks = list(split_text(text, max_length))
    scroll_ratio = 1/len(chunks)
    llm = LlamaModel()
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Summary])
    gbnf_grammar = LlamaGrammar.from_string(gbnf_grammar)
    summary_system_message = f"""
You are an advanced research assistant, specialized in summarizing information extracted from the internet based on a given query. Your task is to analyze the provided text and generate a concise, relevant summary that directly addresses the query.

When presented with a query and accompanying text, follow these steps:

1. Carefully read and understand the query, identifying the key information being requested.

2. Thoroughly analyze the provided text to determine its relevance to the query:
   - If the text is directly relevant, proceed to step 3.
   - If the text is partially relevant, extract and summarize only the relevant portions.
   - If the text is entirely irrelevant, notify the user and provide an explanation.

3. Generate a clear and concise summary that directly addresses the query, focusing on the most pertinent information from the text.

4. Ensure that your summary is well-structured, easy to understand, and free of unnecessary details or redundancies.

5. Follow the formatting instructions provided in the documentation below:

{documentation}

Your goal is to provide accurate, focused, and well-formatted summaries that efficiently address the user's query, saving them time and effort in their research process.
"""
    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, i * scroll_ratio)
            logger.info(f"Scrolling to {i * scroll_ratio * 100}% of the page")
        logger.info(f"Summarizing chunk {i + 1} of {len(chunks)}")
        summary_prompt = summarize_text_prompt(question, chunk)
        summary = llm.chat_completion(
            user_prompt=summary_prompt, system_prompt=summary_system_message,
            grammar=gbnf_grammar, max_tokens=4096, temperature=1.31, top_p=0.14, top_k=49, repeat_penalty=1.17
        )
        try:
            summary = json.loads(summary) 
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response, skipping chunk")
            continue
        summaries.append(Summary(**summary).summary)
    combined_summary = "\nsection\n".join(summaries)
    
    # if the text is too long, summarize the combined summary
    if len(combined_summary) > max_length:
        logger.info(f"Text is too long, summarizing combined text")
        summary_prompt = summarize_text_prompt(question, combined_summary)
        summary = llm.chat_completion(
            user_prompt=summary_prompt, system_prompt=summary_system_message,
            grammar=gbnf_grammar, max_tokens=4096
        )
        summary = json.loads(summary)
        combined_summary = Summary(**summary).summary
    final_summary_ = final_summary_prompt(question, combined_summary)
    final_grammar, documentation = generate_gbnf_grammar_and_documentation([ArticleSummary])
    final_grammar = LlamaGrammar.from_string(final_grammar)
    final_system_message = f"""You are an advanced AI research assistant, tasked with parsing and combining multiple summaries on the same topic into one document. The following is the expected output:\n\n{documentation}"""
    final_summary = llm.chat_completion(
        user_prompt=final_summary_, system_prompt=final_system_message,
        grammar=final_grammar, max_tokens=4096, temperature=1.31, top_p=0.14, top_k=49, repeat_penalty=1.17
    )
    final_summary = json.loads(final_summary)
    return ArticleSummary(**final_summary)    
        
        
    
def scroll_to_percentage(driver: WebDriver, ratio: float) -> None:
    """Scroll to a percentage of the page

    Args:
        driver (WebDriver): The webdriver to use
        ratio (float): The percentage to scroll to

    Raises:
        ValueError: If the ratio is not between 0 and 1
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Percentage should be between 0 and 1")
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {ratio});")
    

    
   
def write_to_file(filename: str, text: str) -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    with open(filename, "w") as file:
        file.write(text)
        

def read_txt_files(directory: str) -> str:
    all_text = ''

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                all_text += file.read() + '\n'

    return all_text       