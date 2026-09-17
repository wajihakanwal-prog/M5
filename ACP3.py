import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

image = Image.open("vintage_photo.jpg").convert("RGB")
mask = Image.open("mask.png").convert("L")

image = image.resize((512, 512))
mask = mask.resize((512, 512))

prompt = """
realistic restoration of a vintage photograph,
natural human details, realistic face,
authentic vintage clothing,
matching original lighting and shadows,
old photographic style, highly detailed,
seamless restoration
"""

negative_prompt = """
blurry, distorted, deformed face, extra fingers,
extra limbs, cartoon, modern clothing,
unrealistic, low quality, artifacts
"""

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

result.save("restored_vintage_photo.png")

display(result)