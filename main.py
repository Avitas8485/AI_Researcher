from research_terminal.research_agent import ResearchAgent

if __name__ == "__main__":
    agent = ResearchAgent()
    question = "Define the term 'quantum computing' and explain its significance."
    agent.run_agent(question)