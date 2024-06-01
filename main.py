import os
from research_terminal.research_agent import ResearchAgent
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

if __name__ == "__main__":
    agent = ResearchAgent()
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
    console = Console()
    console.print(Panel("[bold cyan]Welcome to the Research Terminal![/bold cyan]"))
    while True:
        question = Prompt.ask("Please enter a research question or type 'exit' to quit:", default="Define the term 'quantum computing' and explain its applications.")
        if question == "exit":
            break
        agent.run_agent(question)
    console.print(Panel("[bold green]Question answered![/bold green]"))