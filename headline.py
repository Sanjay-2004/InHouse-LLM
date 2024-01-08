# Working perfect
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline", legacy=False)
model = model.to(device)

article = '''
Coffee is a popular drink made from the roasted beans of Coffea fruits (Coffea arabica, Coffea canephora). It contains caffeine and chlorogenic acid.

The caffeine in coffee works by stimulating the central nervous system (CNS), heart, and muscles. Chlorogenic acid might affect blood vessels and how the body handles blood sugar and metabolism.

People most commonly drink coffee to increase mental alertness. Coffee is also used for diabetes, cancer, heart disease, high blood pressure, dementia, and many other conditions, but there is no good scientific evidence to support most of these uses.

Don't confuse coffee with other caffeine sources, such as green coffee, black tea, and green tea. These are not the same.
'''

text = "headline: " + article

max_len = 256

encoding = tokenizer.encode_plus(text, return_tensors="pt")
input_ids = encoding["input_ids"].to(device)
attention_masks = encoding["attention_mask"].to(device)

beam_outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_masks,
    max_length=64,
    num_beams=3,
    early_stopping=True,
)

result = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
print(result)
