from abc import ABC, abstractmethod, ABCMeta


class SingletonMeta(ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class BaseLLMModel(ABC, metaclass=SingletonMeta):
    @abstractmethod
    def load_model(self, **kwargs):
        pass
    
    @abstractmethod
    def chat_completion(self, user_prompt, **kwargs):
        pass
    