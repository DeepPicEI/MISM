from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("./clip-vit-base-patch32/")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#  processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32/")

img_path = "./ghost.png"
image = Image.open(img_path)

inputs = processor(text=["a photo of a ghost", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image )
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)

