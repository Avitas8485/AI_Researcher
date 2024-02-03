from langchain.document_loaders import PyMuPDFLoader
from langchain.retrievers.arxiv import ArxivRetriever


def scrape_pdf_with_pymupdf(url) -> str:
    """Scrape a pdf with pymupdf

    Args:
        url (str): The url of the pdf to scrape

    Returns:
        str: The text scraped from the pdf
    """
    loader = PyMuPDFLoader(url)
    doc = loader.load()
    return str(doc)


def scrape_pdf_with_arxiv(query) -> str:
    """Scrape a pdf with arxiv
    default document length of 70000 about ~15 pages or None for no limit

    Args:
        query (str): The query to search for

    Returns:
        str: The text scraped from the pdf
    """
    retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None) # type: ignore
    docs = retriever.get_relevant_documents(query=query)
    return docs[0].page_content


if __name__ == '__main__':
    url = "https://arxiv.org/pdf/2103.02559.pdf"
    print("Scraping pdf with pymupdf...")
    print(scrape_pdf_with_pymupdf(url))
    print("Scraping pdf with arxiv...")
    print(scrape_pdf_with_arxiv("pomodoro technique"))