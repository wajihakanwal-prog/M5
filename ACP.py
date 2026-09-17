from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image
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
print ("Available styles:quit' to exit\n")
while True:
    prompt = input("Enter a prompt: ")
    if prompt.lower() in ["quit", "exit"]:
        break
    if not prompt:
        continue
    print("\nChoose a style:")
    print("1.Realistic")
    print("2.Anime")
    print("3.Watercolor")
    print("4.Digital Art")
    print("5.Cinematic")
    print("6. 3DRender")
    style_choice = input("Enter style (1-6): ").strip()
    style = STYLE_PRESETS.get(
        style_choice,
        STYLE_PRESETS["1"]
    )
    negative_prompt = input(
        "\nEnter negative prompt "
    ).strip()

    if not negative_prompt:
        negative_prompt = "blurry, low quality, distorted, deformed"
    print("\nselect image size:")
    print("1. 512 x 512")
    print("2. 768 x 768")
    print("3. 1024 x 1024")
    size_choice = input("Enter size (1-3): ").strip()
    sizes = {
        "1": (512, 512),
        "2": (768, 768),
        "3": (1024, 1024)
    }

    width, height = sizes.get(
        size_choice,
        (512, 512)
    )
    final_prompt = f"""
    {prompt}

    Style:
    {style}

    Avoid:
    {negative_prompt}
    """
    print("Style:", style)
    print("Negative prompt:", negative_prompt)
    print("Size:", width, "x", height)
    image = None
    for model in models:
        try:
            print(f"Trying model: {model}")
            image = client.text_to_image(
                final_prompt,
                model=model
            )
            print(f"Success: Imagegenerated using {model}")
            break
        except Exception as e:
            print("  Model failed.")
            continue
    if image:
      timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
      filename = f"generated_{timestamp}.png"
      image.save(filename)
      print(f"Saved: {filename}")
      image.show()
    else:
        print(
            "\nError: All models failed. " 
        )

