from typing import Generator, Optional
from selenium.webdriver.remote.webdriver import WebDriver 
from research_terminal.llm.llama_model import LlamaModel
import os
from research_terminal.logger.logger import logger


def split_text(text: str, max_length: int = 4000) -> Generator[str, None, None]:
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



def summarize_text(question, text: str, max_length: int = 4000, driver: Optional[WebDriver] = None):
    logger.info(f"Summarizing text with question: {question}")
    summaries = []
    chunks = list(split_text(text, max_length))
    scroll_ratio = 1/len(chunks)
    llm = LlamaModel()
    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, i * scroll_ratio)
            logger.info(f"Scrolling to {i * scroll_ratio * 100}% of the page")
        logger.info(f"Summarizing chunk {i + 1} of {len(chunks)}")
        prompt = create_summary_prompt(chunk, question)
        summary = llm.chat_completion(prompt)
        summaries.append(summary)
        
    combined_summary = "\n".join(summaries)
    final_summary_prompt = f"""Provide a comprehensive explanation of the following query using the information provided. If the question cannot be answered using the text, simply summarize the text itself. Do not include any information that is not in the text.\n\n Query: {question}\n\nInformation{combined_summary}"""
    final_summary = llm.chat_completion(final_summary_prompt)
    logger.info(f"Summarized text with question: {question}")
    
    logger.info(f"Finsihed summarizing text with question: {question}")
    return final_summary


        
        
    
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
    
def create_summary_prompt(chunk, question):
    return f""" Summarize the following text based on the task or question: {question}
    If the question cannot be answered using the text, simply summarize the text itself.
    Do not include any information that is not in the text.
    {chunk}
"""
    
   
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