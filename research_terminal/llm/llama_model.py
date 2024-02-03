from llama_cpp import Llama
from research_terminal.llm.base_llm_model import BaseLLMModel
from contextlib import contextmanager
from research_terminal.logger.logger import logger

class LlamaModel(BaseLLMModel):
    def __init__(self):
        self.model_path = 'C:/Users/avity/Projects/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
        self.llm = self.load_model()
        self.system_prompt = "You are an AI assistant. You are helping a user with a task."
        logger.debug("LlamaModel initialized")
    @contextmanager    
    def load_model(self, **kwargs):
        logger.debug("Loading Llama model")
        llm = Llama(model_path=self.model_path,
                    n_threads=4,
                    n_threads_batch=4,
                    n_ctx=4096,
                    **kwargs)
        
        try:
            yield llm
        finally:
            logger.info("Unloading Llama model")
            llm = None
    
    def chat_completion(self, user_prompt: str, max_retries: int=5, **kwargs)->str:
        logger.info(f"Beginning chat completion")
        retry_count = 0
        while retry_count < max_retries:
            with self.load_model() as self.llm:
                output = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": f"{self.system_prompt}"
                        },
                        {
                            "role": "user",
                            "content": f"{user_prompt}"
                        }
                        ], 
                    temperature=1.31,
                    top_p=0.14,
                    repeat_penalty=1.17,
                    top_k=49,
                    **kwargs
                    )
            
                if output["choices"][0]["message"]["content"] != "": #type: ignore
                    logger.debug(f"Finished chat completion")
                    return output["choices"][0]["message"]["content"] #type: ignore
                else:
                    logger.warning(f"Chat completion returned empty response, retrying")
                    retry_count += 1
        logger.error(f"Chat completion failed after {max_retries} retries")
        raise Exception(f"Chat completion failed after {max_retries} retries")
                
    
    

if __name__ == '__main__':
    llm = LlamaModel()
    prompt = """Provide a comprehensive explanation of the following query using the information provided. If the question cannot be answered using the text, simply summarize the text itself. Do not include any information that is not in the text.\n\n Query:  What is the Pomodoro Technique? \n\nInformation\n\nThe Pomodoro Technique is a time management method that involves breaking work into intervals of 25 minutes, followed by short breaks. It was developed by Francesco Cirillo in the late 1980s to help him focus on his studies and complete assignments more effectively. The technique has since gained popularity among people who struggle with procrastination, distractions, or maintaining sustained concentration during work sessions.\n\nThe Pomodoro Technique is based on four main principles: setting a timer for 25 minutes of focused work, breaking down complex projects into smaller tasks, combining small tasks to maximize efficiency, and not allowing interruptions during the work session. After each 25-minute interval, also known as a pomodoro, individuals take a short break before moving on to the next task or returning to the previous one.\n\nThe technique aims to improve focus and productivity by breaking down large tasks into smaller, more manageable chunks, which can be completed within a set time frame. This approach helps individuals avoid feeling overwhelmed and encourages them to stay focused on their work without getting distracted by other tasks or interruptions.\n\nIn summary, the Pomodoro Technique is a simple yet effective method for managing time and increasing productivity by breaking work into short intervals of focused effort followed by brief breaks.\n The Pomodoro Technique is a time management method that helps individuals break down large tasks into smaller, manageable chunks of time (25 minutes) called pomodoros. It aims to combat procrastination by making tasks less intimidating and combats distractions by focusing on one task at a time during each pomodoro session. Additionally, it helps users become more aware of their time usage by treating time as a positive representation of productivity rather than an abstract concept.\n\n\nThe Pomodoro Technique is a time management method that involves breaking down work into intervals of 25 minutes (called "pomodoros") with short breaks in between. It helps individuals focus on their tasks and improve productivity by providing a clear measurement of finite time and efforts, allowing them to plan their days more efficiently. The technique encourages consistency rather than perfection and can be customized according to individual preferences.\n \n\nThe Pomodoro Technique is a time management method that involves breaking work into intervals, typically 25 minutes in length, separated by short breaks. It aims to increase productivity and focus by allowing for regular breaks and helping individuals stay on task during their work sessions. To implement the technique, users can use an app or physical timer to enforce their pomodoros, schedule tasks in a planner like Todoist, and maintain a clear plan of what they will work on during each session.\n\n\nThe Pomodoro Technique is a time management method that involves breaking tasks into smaller, manageable chunks called "Pomodoros" (usually 25 minutes long) and taking short breaks in between to improve focus and productivity."""
    print(llm.chat_completion(prompt))
        