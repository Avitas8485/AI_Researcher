from research_terminal.llm.llama_model import LlamaModel
from research_terminal.llm.prompts import generate_search_queries_prompt, check_relevance_prompt, answer_question_prompt, research_report_prompt
from research_terminal.llm.llm_parser import queries_to_list, zero_shot_answer
import json
from research_terminal.scraping.web_search import Duckduckgo
from research_terminal.scraping.web_scrape import WebScraper
from research_terminal.scraping.processing.text import summarize_text, write_to_file
from pathvalidate import sanitize_filename
import os
from research_terminal.vector_db.chroma import ChromaDBClient
from research_terminal.logger.logger import logger
from typing import List

class ResearchAgent:
    def __init__(self):
        self.model = LlamaModel()
        self.db = ChromaDBClient()
        self.visited_urls = set()

        
    def create_search_queries(self, question: str)-> List[str]:
        logger.info(f"Creating search queries for question: {question}")
        prompts = generate_search_queries_prompt(question)
        results = self.model.chat_completion(prompts)
        results = queries_to_list(results)
        results = json.dumps(results)
        return json.loads(results)
    
    def search_web(self, query: str, max_links: int=1)-> List[str]:
        ddg = Duckduckgo(query)
        logger.debug(f"Searching the web for query: {query}")
        results = ddg.search()
        search_urls = [result['href'] for result in results if result['href'] not in self.visited_urls]
        new_search_urls = search_urls[:max_links]
        return [self.browse_website(url, query) for url in new_search_urls if url is not None]
    
    
    def browse_website(self, url: str, question: str)-> str:
        logger.info(f"Browsing website: {url}")
        scraper = WebScraper('firefox')
        text = scraper.scrape(url)
        summary = summarize_text(question, text)
        scraper.quit()
        return f"Information from {url}:\n\n{summary}\n\nHyperlinks:\n"
    
    def query_db(self, question: str):
        results = self.db.query_text(question)
        # if the documents returned in the form of [[]], then no results were found
        if results['documents'] == [[]]:
            return None
        return json.dumps(results)
    
    def check_relevance(self, question: str, text: str)-> str:
        prompt = check_relevance_prompt(question, text)
        result = self.model.chat_completion(prompt)
        relevance = zero_shot_answer(result)
        return relevance
    
    def process_db_results(self, question: str, db_results: str):
        logger.info(f"Checking relevance of database results for question: {question}")
        relevance = self.check_relevance(question, db_results)
        if relevance == "confirmation":
            logger.info(f"Database results are relevant for question: {question}")
            prompt = answer_question_prompt(question, db_results)
            result = self.model.chat_completion(prompt)
            return result
        else:
            logger.info(f"Database results not relevant for question: {question}")
            return None
     
    def process_web_search(self, question: str):
        logger.info(f"Searching the web for question: {question}")
        queries = self.create_search_queries(question)
        search_results = [self.search_web(query) for query in queries]
        logger.debug(f"Search completed for question: {question} saving results...")
        research_info = '\n\n'.join(result for search_result in search_results for result in search_result)     
        research_report = research_report_prompt(queries, research_info)
        result = self.model.chat_completion(research_report)
        logger.debug(f"Presenting research information for question: {question}")
        print(result)
        os.makedirs('./results', exist_ok=True)
        write_to_file(f"./results/{sanitize_filename(question)}.txt", result)
        self.db.add_text(result)
        return result
       
    def run_agent(self, question: str):
        logger.info(f"Running research agent for question: {question}")
        logger.info(f"Checking database for question: {question}")
        db_results = self.query_db(question)
        if db_results:
            logger.info(f"Database results found for question: {question}")
            result = self.process_db_results(question, db_results)
            if result:
                print(result)
                return result
        return self.process_web_search(question)


if __name__ == "__main__":
    agent = ResearchAgent()
    question = "What is the Pomodoro Technique?"
    agent.run_agent(question)