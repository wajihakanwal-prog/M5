from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load the AI image-captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image_path):
    """Generate a caption for one image."""

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


def main():
    report = []

    print("=== Multiple Image Captioning ===")

    while True:
        image_path = input(
            "\nEnter image path (or type 'done' to finish): "
        )

        if image_path.lower() == "done":
            break

        try:
            caption = generate_caption(image_path)

            print("Caption:", caption)

            report.append({
                "image": image_path,
                "caption": caption
            })

        except FileNotFoundError:
            print("Error: Image file not found.")

        except Exception as e:
            print("Error:", e)

    # Save captions to a report
    with open("caption_report.txt", "w", encoding="utf-8") as file:

        file.write("IMAGE CAPTION REPORT\n")
        file.write("=" * 40 + "\n\n")

        for item in report:
            file.write(f"Image: {item['image']}\n")
            file.write(f"Caption: {item['caption']}\n")
            file.write("-" * 40 + "\n")

    print("\nReport saved as caption_report.txt")


if __name__ == "__main__":
    main()