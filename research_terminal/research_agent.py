from research_terminal.llm.llama_model import LlamaModel
from research_terminal.llm.prompts import generate_search_queries_prompt,  answer_question_prompt, research_report_prompt, generate_report_prompt
from research_terminal.llm.llm_parser import check_relevance
import json
from research_terminal.llm.grammar.pydantic_models import Summary
from research_terminal.scraping.web_search import Duckduckgo
from research_terminal.scraping.web_scrape import WebScraper
from research_terminal.scraping.processing.text import summarize_text, write_to_file
from research_terminal.llm.grammar.pydantic_models_to_grammar import generate_gbnf_grammar_and_documentation
from llama_cpp.llama_grammar import LlamaGrammar
from research_terminal.llm.grammar.pydantic_models import SearchQueries, ResearchReport
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
        

    
    def generate_search_queries(self, question: str, num_queries: int = 3) -> SearchQueries:
        """Generates the search queries for the given question.
        Args:
            question (str): The question to generate the search queries for
        Returns:
            SearchQueries: The search queries for the given question
            """
        search_query_prompt = generate_search_queries_prompt(question, num_queries)
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([SearchQueries])
        gbnf_grammar = LlamaGrammar.from_string(gbnf_grammar)
        search_query_system_message = """You are an advanced AI research assistant, tasked with creating search queries in JSON format to find information on a given prompt. The following is the expected output:\n\n""" + documentation

        search_queries = self.model.chat_completion(
            user_prompt=search_query_prompt, system_prompt=search_query_system_message,
            grammar=gbnf_grammar, max_tokens=1024
        )

        search_queries = json.loads(search_queries) #type: ignore
        return SearchQueries(**search_queries)
    
    
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
        self.visited_urls.add(url)
        scraper.quit()
        return f"""
    Article Summary:
        Title: {summary.title}
        Source: {url}
        Question: {summary.question}
        Main Ideas: {summary.main_ides}
        Chunk Summaries: 
        {self.beautify_chunk_summaries(summary.chunk_summaries)}
            
        Strengths: {summary.strengths}
        Weaknesses: {summary.weaknesses}
        Conclusion: {summary.conclusion}
    
    """
    # chunk summaries is a list of Summary objects
    def beautify_chunk_summaries(self, chunk_summaries: List[Summary])-> str:
        return '\n'.join(f"Question: {summary.question}\nSummary: {summary.summary}\nRelevance: {summary.relevance}" for summary in chunk_summaries)
    
    
    def query_db(self, question: str):
        results = self.db.query_text(question)
        # if the documents returned in the form of [[]], then no results were found
        if results['documents'] == [[]]:
            return None
        return json.dumps(results['documents'])
    
    
        
    
    def process_db_results(self, question: str, db_results: str):
        logger.info(f"Checking relevance of database results for question: {question}")
        relevance = check_relevance(question, db_results)
        if relevance.relevance == 'yes':
            logger.info(f"Database results are relevant for question: {question}")
            prompt = answer_question_prompt(question, db_results)
            result = self.model.chat_completion(system_prompt=self.model.system_prompt, user_prompt=prompt)
            return result
        else:
            logger.info(f"Database results not relevant for question: {question}")
            return None
    def generate_report(self, queries: List[str], research_info: str)-> ResearchReport:
        logger.info(f"Generating research report")
        report_prompt = research_report_prompt(queries, research_info)
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([ResearchReport])
        gbnf_grammar = LlamaGrammar.from_string(gbnf_grammar)
        report_system_message = generate_report_prompt(documentation=documentation)
        report = self.model.chat_completion(
            user_prompt=report_prompt, system_prompt=report_system_message,
            grammar=gbnf_grammar, max_tokens=4096, temperature=1.31, top_p=0.14, top_k=49, repeat_penalty=1.17
        )
        report = json.loads(report)
        return ResearchReport(**report)
    
    def beautify_report(self, report: ResearchReport) -> str:
        return (
            f"Report on '{report.original_question}'\n\n"
            f"{report.title}\n\n"
            f"Queries: {', '.join(report.queries)}\n\n"
            f"{report.executive_summary}\n\n"
            f"{report.introduction.title}\n\n"
            f"{report.introduction.content}\n\n"
            + "\n\n".join(f"{section.title}\n\n{section.content}" for section in report.main_body) + "\n\n"
            f"{report.conclusion.title}\n\n{report.conclusion.content}\n\n"
            f"References:\n\n"
            + "\n\n".join(f"{reference.title}\n{reference.url}" for reference in report.references)
    )

    def process_web_search(self, question: str):
        logger.info(f"Searching the web for question: {question}")
        search_results = [self.search_web(question, 3)]
        logger.debug(f"Search completed for question: {question} saving results...")
        research_info = '\n\n'.join(result for search_result in search_results for result in search_result)     
        result = self.generate_report([question], research_info)
        result = self.beautify_report(result)
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

    
    
