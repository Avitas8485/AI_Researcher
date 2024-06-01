from llama_cpp import Llama
from research_terminal.llm.base_llm_model import BaseLLMModel

from research_terminal.logger.logger import logger


class LlamaModel(BaseLLMModel):
    def __init__(self):
        self.model_path = 'C:/Users/avity/Projects/models/llm/stablelm-zephyr-3b.Q4_K_M.gguf'
        self.llm = self.load_model()
        self.system_prompt = "You are an AI assistant. You are helping a user with a task."
        logger.debug("LlamaModel initialized")

    def load_model(self, **kwargs)-> Llama:
        """Load the Llama model

        Returns:
            Llama: The Llama model
        """
        logger.debug("Loading Llama model")
        llm = Llama(model_path=self.model_path,
                    n_threads=5,
                    n_threads_batch=5,
                    n_gpu_layers=30,
                    flash_attn=True,
                    n_ctx=4096, 
                    **kwargs)
        
        return llm
    
    def chat_completion(self, system_prompt: str, user_prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Complete a chat prompt
            Args:
                system_prompt (str): The system prompt
                user_prompt (str): The user prompt
                max_retries (int): The maximum number of retries
            Returns:
                str: The completion of the chat prompt"""
        logger.info(f"Beginning chat completion")
        retry_count = 0
        while retry_count < max_retries:
            output = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"{user_prompt}"
                    }
                    ],
                **kwargs
                )
            if "choices" in output and len(output["choices"]) > 0: #type: ignore
                if output["choices"][0]["message"]["content"] != "": #type: ignore
                    logger.debug(f"Finished chat completion")
                    return output["choices"][0]["message"]["content"] #type: ignore
                else:
                    logger.warning(f"Chat completion returned empty response, retrying")
                    retry_count += 1
        logger.error(f"Chat completion failed after {max_retries} retries")
        raise Exception(f"Chat completion failed after {max_retries} retries")
                
    
    