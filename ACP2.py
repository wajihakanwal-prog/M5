from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
from config import HF_API_KEY
models = [
    "ByteDance/SDXL-Lightning",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5"
]
STYLE_PRESETS = {
    "1": "realistic photography, highly detailed, natural lighting",
    "2": "anime style, vibrant colors, detailed anime artwork",
    "3": "watercolor painting, soft colors, artistic brush strokes",
    "4": "digital art, highly detailed, professional digital artwork",
    "5": "cinematic style, dramatic lighting, movie-like composition",
    "6": "3D render, realistic textures, high quality 3D artwork"
}
client = InferenceClient(api_key=HF_API_KEY)

print(f"Primary model: {models[0]}")
print("Type 'quit' to exit\n")

while True:
    prompt = input("Enter a prompt: ").strip()

    if prompt.lower() in ["quit", "exit", "q"]:
        break

    if not prompt:
        continue

    print("\nChoose a style:")
    print("1. Realistic")
    print("2. Anime")
    print("3. Watercolor")
    print("4. Digital Art")
    print("5. Cinematic")
    print("6. 3D Render")

    style_choice = input("Enter style (1-6): ").strip()

    style = STYLE_PRESETS.get(
        style_choice,
        STYLE_PRESETS["1"]
    )

    negative_prompt = input(
        "\nEnter negative prompt: "
    ).strip()

    if not negative_prompt:
        negative_prompt = "blurry, low quality, distorted, deformed"

    final_prompt = f"""
{prompt}

Style:
{style}

Avoid:
{negative_prompt}
"""

    print("\nGenerating image...")

    image = None

    for model in models:
        try:
            print(f"Trying model: {model}")

            image = client.text_to_image(
                final_prompt,
                model=model
            )

            print(f"Success: Image generated using {model}")
            break

        except Exception:
            print("Model failed. Trying next...")
            continue

    if image:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        original_filename = f"original_{timestamp}.png"
        image.save(original_filename)

        daylight = ImageEnhance.Brightness(image).enhance(1.35)
        daylight = ImageEnhance.Contrast(daylight).enhance(1.10)
        daylight = daylight.filter(
            ImageFilter.GaussianBlur(radius=1)
        )

        daylight_filename = f"daylight_{timestamp}.png"
        daylight.save(daylight_filename)

        night = ImageEnhance.Brightness(image).enhance(0.75)
        night = ImageEnhance.Contrast(night).enhance(1.40)
        night = night.filter(
            ImageFilter.GaussianBlur(radius=0.7)
        )

        night_filename = f"night_mood_{timestamp}.png"
        night.save(night_filename)

        print(f"\nOriginal saved: {original_filename}")
        print(f"Daylight Edition saved: {daylight_filename}")
        print(f"Night Mood saved: {night_filename}")

        image.show()
        daylight.show()
        night.show()

    else:
        print("\nError: All models failed. Check your API key.\n")

