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
    "meta-llama/Llama-3.2-11B-Vision-Instruct:sambanova"
]

TEXT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
    "mistralai/Mistral-7B-Instruct-v0.3:together"
]


def data_url(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    return "data:image/png;base64," + data


def query_api(payload):
    try:
        response = requests.post(
            ROUTER_URL,
            headers=HEADERS,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        return None


def extract_text(data):
    try:
        return data["choices"][0]["message"]["content"]
    except:
        return ""


def run_model(models, messages, max_tokens=200):

    for model in models:

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4
        }

        data = query_api(payload)

        text = extract_text(data)

        if text:
            return text

    return ""


def generate_caption(image_path):

    image = data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one short sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            ]
        }
    ]

    return run_model(
        VISION_MODELS,
        messages,
        100
    )


def generate_description(caption):

    prompt = f"""
Based on this image caption:

{caption}

Write a detailed description of the image in about 30 words.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return run_model(
        TEXT_MODELS,
        messages,
        150
    )


def generate_summary(description):

    prompt = f"""
Summarize the following image description in about 50 words:

{description}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return run_model(
        TEXT_MODELS,
        messages,
        150
    )


def create_image_prompt(description):

    prompt = f"""
Convert this image description into a creative text-to-image prompt.

Description:
{description}

Return only the image generation prompt.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return run_model(
        TEXT_MODELS,
        messages,
        150
    )


def main():

    print(Fore.CYAN + "\nAI IMAGE WORKFLOW")

    image_path = input(
        "Enter image path: "
    ).strip()

    try:
        image = Image.open(image_path)
        image.verify()
    except:
        print(Fore.RED + "Invalid image.")
        return

    print(Fore.YELLOW + "\nGenerating caption...")

    caption = generate_caption(image_path)

    if not caption:
        print(Fore.RED + "Caption generation failed.")
        return

    print(Fore.GREEN + "\nCAPTION:")
    print(caption)

    print(Fore.YELLOW + "\nGenerating description...")

    description = generate_description(caption)

    print(Fore.GREEN + "\nDESCRIPTION:")
    print(description)

    print(Fore.YELLOW + "\nGenerating summary...")

    summary = generate_summary(description)

    print(Fore.GREEN + "\nSUMMARY:")
    print(summary)

    print(Fore.YELLOW + "\nCreating text-to-image prompt...")

    image_prompt = create_image_prompt(description)

    print(Fore.GREEN + "\nTEXT-TO-IMAGE PROMPT:")
    print(image_prompt)

    with open(
        "ai_image_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("AI IMAGE REPORT\n")
        file.write("=" * 40 + "\n\n")

        file.write("CAPTION:\n")
        file.write(caption + "\n\n")

        file.write("DESCRIPTION:\n")
        file.write(description + "\n\n")

        file.write("SUMMARY:\n")
        file.write(summary + "\n\n")

        file.write("TEXT-TO-IMAGE PROMPT:\n")
        file.write(image_prompt + "\n")

    print(
        Fore.CYAN +
        "\nReport saved as ai_image_report.txt"
    )


if __name__ == "__main__":
    main()