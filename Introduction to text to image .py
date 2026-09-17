from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image
from config import HF_API_KEY
models = [
    "ByteDance/SDXL-Lightning",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5", 
]
client = InferenceClient(api_key=HF_API_KEY)
print(f"model: {models[0]}")
print("Type 'q' to exit")
while True:
    prompt = input("prompt: ")
    if prompt.lower() in ["q", "exit"]:
        break
    if not prompt:
        continue
    print("generating image...")
    image = None
    for model in models:
        try:
            image = client.text_to_image(prompt, model=model)
            break  
        except Exception:
            print(f"next model: {model}")
            continue
    if image:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        image.save(filename)
        print(f"saved: {filename}")
        image.show()
        print()
    else:
        print("Error: failed to generate image.\n")


