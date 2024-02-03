from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from research_terminal.logger.logger import logger
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def queries_to_list(queries: str)-> list[str]:
    # I don't want to rely on llms to produce a list of queries so I decided to use t5 to parse the queries
    # I'm too lazy to use other method, this seemed the easiest. feel free to implement a better method
    # todo: implement a better method
    """ Converts the given queries to a list of strings."""
    logger.info(f"Converting queries to list: {queries}")
    prompt = "Extract only the statements after each number"
    input_text = f"{prompt}\n {queries}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1000)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f"Decoded queries: {decoded}")
    output = re.split(r"\d+\.", decoded)[1:]    
    # there are some cases where the output has more than one quotation mark, so we need to remove them
    return [re.sub(r'\"', '', query) for query in output]
    

def zero_shot_answer(text: str)-> str:
    # since llms are not known for giving a straight answer, we can use the zero-shot classifier to get a straight answer based on the llm's output
    model_name = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_name)
    candidate_labels = ["confirmation", "rejection"]
    result = classifier(text, candidate_labels)
    return result["labels"][0] #type: ignore


if __name__ == "__main__":
    queries = """1. "Pomodoro Technique definition and explanation"
                2. "How does the Pomodoro Technique work?"      
                3. "Benefits of using the Pomodoro Technique for productivity"""
    print(queries_to_list(queries))
    