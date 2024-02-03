from duckduckgo_search import DDGS
from typing import Generator, Dict
from research_terminal.logger.logger import logger


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query):
        self.ddg = DDGS()
        self.query = query
        logger.info(f"Duckduckgo initialized with query: {query}")

    def search(self, max_results=5)-> Generator[Dict[str, str | None], None, None]:
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        logger.info(f"Searching Duckduckgo for query: {self.query}")
        ddgs_gen = self.ddg.text(self.query, region='wt-wt', max_results=max_results)
        return ddgs_gen
    
if __name__ == '__main__':
    ddg = Duckduckgo("pomodoro technique")
    for result in ddg.search():
        print(result)