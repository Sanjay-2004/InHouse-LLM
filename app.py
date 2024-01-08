from flask import Flask, render_template, request
import subprocess
import os
from PIL import Image
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlipProcessor, BlipForConditionalGeneration, DistilBertTokenizer, DistilBertForQuestionAnswering, BartForConditionalGeneration, BartTokenizer
import torch

app = Flask(__name__)

# Load pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# T5 model for headline generation
headline_model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
headline_tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline", legacy=False)
headline_model = headline_model.to(device)

# Image captioning model
img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
img_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Q&A model
qna_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
qna_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qna_model.to(device)

# Summary model
summary_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summary_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summary_model = summary_model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.form.get('text')
    file = request.files.get('file')
    questions_input = request.form.get('questionsInput')  # Get questions input

    # Save the uploaded file
    if file:
        file_path = 'uploads/' + file.filename
        file.save(file_path)
    else:
        file_path = None

    # Choose the appropriate Python script based on the action
    action = request.form.get('action')
    if action == 'headline':
        result = generate_headline(text)
    elif action == 'summarize':
        result = generate_summary(text)
    elif action == 'generateQuestions':
        result = generate_auto_questions(text)
    elif action == 'img2txt':
        result = generate_image_caption(file_path) if file_path else 'No file provided for image captioning.'
    else:
        result = 'Invalid action'

    return result

def generate_auto_questions(text):
    # You can implement your logic here to generate questions automatically
    # For now, let's generate a dummy list of questions
    dummy_questions = ["What is the main idea of the text?", "Who are the key characters?", "What is the setting?"]
    return '\n'.join(dummy_questions)

def generate_headline(text):
    max_len = 256

    encoding = headline_tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = headline_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=64,
        num_beams=3,
        early_stopping=True,
    )

    result = headline_tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
    return result

def generate_image_caption(img_path):
    raw_image = Image.open(img_path).convert('RGB')

    # unconditional image captioning
    inputs = img_processor(raw_image, return_tensors="pt").to(device)

    # Set max_length explicitly to remove the warning
    out = img_model.generate(**inputs, max_length=50)  # Adjust the value according to your needs
    return img_processor.decode(out[0], skip_special_tokens=True)

def process_text_with_questions(text, questions_input):
    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]

    text = text
    answers = []
    for question in questions:
        answer = answer_question(question, text)
        answers.append({'question': question, 'answer': answer})

    return answers

def generate_qna(questions_input, text):
    answers = process_text_with_questions(text, questions_input)
    result = ''
    for answer in answers:
        result += f"Question: {answer['question']}<br>Answer: {answer['answer']}<br><br>"

    return result

def answer_question(question, text):
    question_tokens = qna_tokenizer.tokenize(question)
    text_tokens = qna_tokenizer.tokenize(text)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + text_tokens + ['[SEP]']
    input_ids = qna_tokenizer.convert_tokens_to_ids(tokens)

    inputs = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = qna_model(inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    predict_answer_tokens = input_ids[answer_start_index: answer_end_index + 1]
    predict_answer = qna_tokenizer.decode(predict_answer_tokens)

    return predict_answer

def generate_summary(input_text):
    inputs = summary_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  
    summary_ids = summary_model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
