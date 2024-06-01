from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SearchQueries(BaseModel):
    """Represents the search queries on a given prompt."""
    queries: list[str] = Field(..., description="What search queries would you use to find information on the given prompt? The search queries should be independent of each other.")
    chain_of_thought: str = Field(..., description="What is the chain of thought that led you to choose these search queries?")
    

class Relevant(Enum):
    """Represents the relevance of the database information to the query."""
    YES = "yes"
    NO = "no"
    
class CheckRelevance(BaseModel):
    """Represents the relevance of the database information to the query."""
    relevance: Relevant = Field(..., description="Is the database information relevant to the query?")
    rationale: str = Field(..., description="What is your rationale for your decision?")
    


class Summary(BaseModel):
    """Represent a summary of the given text with respect to the question."""
    question: str = Field(..., description="The prompt or question to summarize the text with respect to")   
    summary: str = Field(..., description="The summary of the text with respect to the question, be as detailed as possible and include any factual information such as numbers, stats, quotes, etc if available")
    relevance: Relevant = Field(..., description="Is the summary or chunk relevant to the question?")
    
class ArticleSummary(BaseModel):
    """Represent a summary of the given text with respect to the question."""
    title: str = Field(..., description="The title of the article to summarize")
    question: str = Field(..., description="The prompt or question to summarize the text with respect to")
    main_ides: str = Field(..., description="The main ideas of the article")
    chunk_summaries: List[Summary] = Field(..., description="The summaries collected from each chunk of the article")
    strengths: Optional[str] = Field(None, description="What are the strengths of the article?")
    weaknesses: Optional[str] = Field(None, description="What are the weaknesses of the article?")
    conclusion: Optional[str] = Field(None, description="What is the conclusion can you draw from this research? What are the implications of this research and what are the next steps?")
    


class Reference(BaseModel):
    """Represents a reference to a source used in the research report."""
    title: str = Field(..., description="The title of the referenced source")
    url: str = Field(..., description="The URL where the source can be accessed")

class Section(BaseModel):
    """Represents a section of the research report, containing a title and content."""
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The detailed content of the section")

class ResearchReport(BaseModel):
    """Represents a comprehensive research report generated from various queries and sources."""
    title: str = Field(..., description="The title of the research report")
    original_question: str = Field(..., description="The original research question that the report aims to answer")
    queries: List[str] = Field(..., description="The list of queries used to gather information for the report")
    executive_summary: str = Field(..., description="A brief summary of the key findings and conclusions of the report")
    introduction: Section = Field(..., description="The introduction section, providing background and context for the report")
    main_body: List[Section] = Field(..., description="The main body of the report, divided into multiple sections covering different aspects of the research")
    conclusion: Section = Field(..., description="The conclusion section, summarizing the findings and implications of the research")
    references: List[Reference] = Field(..., description="A list of all sources referenced in the report, including titles and URLs")
    

    
  