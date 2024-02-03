from abc import ABC, abstractmethod

class BaseLLMModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        pass
    
    @abstractmethod
    def chat_completion(self, user_prompt, **kwargs):
        pass
    