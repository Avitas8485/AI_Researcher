from research_terminal.research_agent import ResearchAgent

if __name__ == "__main__":
    agent = ResearchAgent()
    question = "What is the Pomodoro Technique?"
    agent.run_agent(question)