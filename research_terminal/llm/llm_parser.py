from research_terminal.logger.logger import logger
from research_terminal.llm.prompts import check_relevance_prompt
from research_terminal.llm.grammar.pydantic_models import CheckRelevance
from research_terminal.llm.grammar.pydantic_models_to_grammar import generate_gbnf_grammar_and_documentation
from llama_cpp.llama_grammar import LlamaGrammar
import json
from research_terminal.llm.llama_model import LlamaModel


def check_relevance(question: str, text: str) -> CheckRelevance:
    """Generates the check relevance for the given question and text.
    Args:
        question (str): The question to generate the check relevance for
        text (str): The text to generate the check relevance for
    Returns:
        CheckRelevance: The check relevance for the given question and text
    """
    llm = LlamaModel()
    check_relevance = check_relevance_prompt(question, text)
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([CheckRelevance])
    gbnf_grammar = LlamaGrammar.from_string(gbnf_grammar)
    check_relevance_system_message = """You are an advanced AI research assistant, tasked with determining whether the information provided is relevant to the query. The following is the expected output:\n\n""" + documentation
    relevance = llm.chat_completion(
        user_prompt=check_relevance, system_prompt=check_relevance_system_message,
        grammar=gbnf_grammar, max_tokens=512
    )
    relevance = json.loads(relevance) #type: ignore
    return CheckRelevance(**relevance)

