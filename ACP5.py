from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

image = Image.open("image.jpg").convert("RGB")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(output[0], skip_special_tokens=True)

print("Basic Caption:", caption)

generator = pipeline("text-generation", model="gpt2")

prompt = f"Expand this image caption into a detailed description: {caption}"

result = generator(
    prompt,
    max_new_tokens=60,
    num_return_sequences=1,
    do_sample=True
)

print("\nExpanded Caption:")
print(result[0]["generated_text"])