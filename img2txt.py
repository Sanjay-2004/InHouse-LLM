#Works perfect
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

img_path = r'C:\Users\ASUS\Desktop\works\Screenshot_20230226_105221.png'
raw_image = Image.open(img_path).convert('RGB')

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to(device)

# Set max_length explicitly to remove the warning
out = model.generate(**inputs, max_length=50)  # Adjust the value according to your needs
print(processor.decode(out[0], skip_special_tokens=True))
