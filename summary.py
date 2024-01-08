#Working perfect
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = model.to(device)

# Input text
input_text = '''Once upon a time, in a forest, there lived a speedy and boastful rabbit. He was known throughout the forest for his swiftness and his habit of making fun of the slower animals. One day, he came across a slow-moving tortoise who was plodding along, carrying a heavy shell on his back.

The rabbit couldn't resist mocking the tortoise for his sluggish pace. "Why do you move so slowly?" he taunted. "You'll never get anywhere at this rate. I can run circles around you without even breaking a sweat."

The tortoise, although slow, was wise and calm. He replied, "Well, my friend, perhaps I am slow, but I am steady. I may not be as fast as you, but I have a determination that will take me far. How about a race to settle this?"

The rabbit was taken aback by the tortoise's challenge. He thought it was a joke and agreed to the race, thinking he could easily win. They chose a course, and the race began.

The rabbit sprinted ahead and quickly left the tortoise far behind. He was so confident of victory that he decided to take a nap under a tree, believing there was no chance the tortoise could catch up.

Meanwhile, the tortoise continued to plod along steadily. He knew he couldn't match the rabbit's speed, but he didn't stop or get distracted. He just kept moving forward, one slow step at a time.

When the rabbit woke up and saw that the tortoise was nowhere in sight, he thought he had all the time in the world. He decided to take his time, nibbling on some grass and chatting with other animals along the way.

As the rabbit dawdled, the tortoise kept moving steadily toward the finish line. Slowly but surely, he closed the gap between them.

When the rabbit finally saw the tortoise approaching the finish line, he panicked and raced ahead with all his might. But it was too late. The tortoise had reached the finish line first, winning the race.'''

# Tokenize and generate summary
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device
summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and print the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
