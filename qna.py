#Working perfect
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
model.to(device)

def answer_question(question, text):
    inputs = tokenizer(question, text, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predict_answer = tokenizer.decode(predict_answer_tokens)
    
    return predict_answer

# List of questions
questions = [
    "What were Marie Curie's significant contributions to science?",
    "Where was Marie Curie born?",
    "When did Marie Curie move to Paris?",
    # Add more questions as needed
]

# Given text
text = "Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity. Born in Warsaw, Poland in 1867, she later moved to Paris and became the first woman to win a Nobel Prize. Marie Curie's notable achievements include the discovery of the elements polonium and radium. Her work laid the foundation for the development of X-ray machines, and her contributions to science have left a lasting legacy."

# Answer each question
for question in questions:
    answer = answer_question(question, text)
    print(f"Question: {question}\nAnswer: {answer}\n")
