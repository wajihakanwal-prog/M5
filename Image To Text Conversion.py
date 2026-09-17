from config import HF_API_KEY
import requests, base64, os, re, time
from PIL import Image
from colorama import init, Fore, Style
init(autoreset=True)

Router_url ="https://router.huggingface.co/v1/chat/completions"
Headers={"Authorization":f"Bearer {HF_API_KEY}","Content-Type":"application/json"}

Vision_models = [
    "moonshotai/Kimi-K2.6:novita",                               # 2.6B
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova",   # 17B
    "meta-llama/Llama-3.2-11B-Vision-Instruct:sambanova"
]
Text_models = [
    "Qwen/Qwen2.5-7B-Instruct:together"
    "Qwen/Qwen2.5-14B-Instruct:together"
    "Qwen/Qwen2.5-32B-Instruct:together"
    "mistralai/Mistral-7B-Instruct-v0.3:together"
    "mistralai/Mixtral-8x7B-Instruct-v0.1:together"
]

def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

def query_hf_api(payload: dict):
    try:
        response = requests.post(Router_url, headers=Headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def _extract_text(data) -> str:
    return data["choices"][0]["message"]["content"]
def run_models(models, messages, max_tokens=160, temperature=0.3):
    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = query_hf_api(payload)
        text = _extract_text(data)
        yield text
def words(text: str):
    return text.split()
def _exact_n_words(text: str, n: int) -> str:
    return " ".join(words(text)[:n])
def _ensure_sentence_end(text: str) -> str:
    if text[-1] not in ".!?":
        return text + "."
    return text
def generate_text(prompt: str, max_new_tokens: int = 220) -> str:
    messages = [{"role": "user", "content": prompt}]
    for text in run_models(TEXT_MODELS, messages, max_new_tokens=max_new_tokens):
        return _ensure_sentence_end(text)
    return ""
def generate_exact_sentence(prompt: str, n_words: int, max_new_tokens: int, tries: int = 6) -> str: 
        raise Exception("No text found")
        return _ensure_sentence_end(text)
def get_basic_caption(image_path: str) -> str:
    messages = [{"role": "user", "content": f"Describe this image: {_data_url(image_path)}"}]
    for text in run_models(VISION_MODELS, messages, max_new_tokens=160):        

        return _exact_n_words(text, n_words=10)
    return ""
def print_menu():
    print("select the output type")
    print("1. caption 5 words")
    print("2. Description 30 words")
    print("3. Summary 50 words")
    print("4. Exit")
def main():
    Image_path=input("Enter the image path")
    try:
        Image.open(Image_path)
    except:
        print("Invalid Image Path")
        return
    basic_caption=get_basic_caption(Image_path)
    print_menu()
    choice=int(input("Enter your choice"))
    if choice==1:
        print(generate_exact_sentence(basic_caption,5,220))
    elif choice==2:
        print(generate_exact_sentence(basic_caption,30,220))
    elif choice==3:
        print(generate_exact_sentence(basic_caption,50,220))
    elif choice==4:
        return
    else:
        print("Invalid choice")
        return

if __name__==__main__:
    main()
    
 