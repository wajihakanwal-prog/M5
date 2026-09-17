from config import HF_API_KEY
import requests
import base64
import time
from PIL import Image
from colorama import init, Fore

init(autoreset=True)

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

VISION_MODELS = [
    "moonshotai/Kimi-K2.6:novita",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova",
    "meta-llama/Llama-3.2-11B-Vision-Instruct:sambanova"
]

TEXT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
    "Qwen/Qwen2.5-14B-Instruct:together",
    "Qwen/Qwen2.5-32B-Instruct:together",
    "mistralai/Mistral-7B-Instruct-v0.3:together",
    "mistralai/Mixtral-8x7B-Instruct-v0.1:together"
]

def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + image_data

def query_hf_api(payload: dict):
    try:
        response = requests.post(
            ROUTER_URL,
            headers=HEADERS,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"API Error: {e}")
        return None
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        return None

def _extract_text(data) -> str:
    if not data:
        return ""

    try:
        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])

            return " ".join(text_parts)

        return str(content)

    except (KeyError, IndexError, TypeError):
        return ""

def words(text: str):
    return text.split()

def _words(text: str):
    return words(text)

def _exact_n_words(text: str, n: int) -> str:
    return " ".join(words(text)[:n])

def _ensure_sentence_end(text: str) -> str:
    text = text.strip()

    if not text:
        return ""

    if text[-1] not in ".!?":
        text += "."

    return text

def run_models(
    models,
    messages,
    max_tokens=160,
    temperature=0.3
):
    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        data = query_hf_api(payload)
        text = _extract_text(data)

        if text:
            yield text

def generate_text(
    prompt: str,
    max_new_tokens: int = 220
) -> str:

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    for text in run_models(
        TEXT_MODELS,
        messages,
        max_tokens=max_new_tokens,
        temperature=0.4
    ):
        return _ensure_sentence_end(text)

    return ""

def generate_exact_sentence(
    prompt: str,
    n_words: int,
    max_new_tokens: int = 220,
    tries: int = 6
) -> str:

    last = ""

    for _ in range(tries):

        current_prompt = (
            f"{prompt}\n\n"
            f"Write a clear description using at least {n_words} words. "
            f"Do not use bullet points. Write it as a single sentence."
        )

        last = generate_text(
            current_prompt,
            max_new_tokens=max_new_tokens
        )

        if len(_words(last)) >= n_words:
            return _ensure_sentence_end(
                _exact_n_words(last, n_words)
            )

        time.sleep(0.2)

    if last:
        return _ensure_sentence_end(
            _exact_n_words(
                last,
                min(n_words, len(_words(last)))
            )
        )

    return ""

def get_basic_caption(image_path: str) -> str:

    image_data = _data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]
        }
    ]

    for text in run_models(
        VISION_MODELS,
        messages,
        max_tokens=160,
        temperature=0.3
    ):
        return _exact_n_words(text, 10)

    return ""

def print_menu():
    print()
    print(Fore.CYAN + "=" * 40)
    print(Fore.CYAN + "       IMAGE CAPTIONING MENU")
    print(Fore.CYAN + "=" * 40)
    print("1. Caption - 5 words")
    print("2. Description - 30 words")
    print("3. Summary - 50 words")
    print("4. Exit")
    print(Fore.CYAN + "=" * 40)

def main():

    print(Fore.GREEN + "\nAI IMAGE CAPTIONING PROGRAM")

    image_path = input(
        "\nEnter the image path: "
    ).strip()

    try:
        image = Image.open(image_path)
        image.verify()
    except Exception:
        print(
            Fore.RED +
            "Invalid image path or unsupported image file."
        )
        return

    print(
        Fore.YELLOW +
        "\nGenerating basic image caption..."
    )

    basic_caption = get_basic_caption(image_path)

    if not basic_caption:
        print(
            Fore.RED +
            "Could not generate an image caption."
        )
        return

    print(
        Fore.GREEN +
        f"\nBasic Caption: {basic_caption}"
    )

    print_menu()

    try:
        choice = int(
            input("Enter your choice: ")
        )
    except ValueError:
        print(
            Fore.RED +
            "Please enter a valid number."
        )
        return

    if choice == 1:

        result = generate_exact_sentence(
            basic_caption,
            5,
            220
        )

        print(
            Fore.GREEN +
            f"\n5-Word Caption:\n{result}"
        )

    elif choice == 2:

        result = generate_exact_sentence(
            basic_caption,
            30,
            220
        )

        print(
            Fore.GREEN +
            f"\n30-Word Description:\n{result}"
        )

    elif choice == 3:

        result = generate_exact_sentence(
            basic_caption,
            50,
            220
        )

        print(
            Fore.GREEN +
            f"\n50-Word Summary:\n{result}"
        )

    elif choice == 4:

        print(
            Fore.YELLOW +
            "Exiting program..."
        )
        return

    else:

        print(
            Fore.RED +
            "Invalid choice."
        )

if __name__ == "__main__":
    main()