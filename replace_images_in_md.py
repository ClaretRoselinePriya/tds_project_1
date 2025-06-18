import re
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Cache to avoid repeated downloads
caption_cache = {}

def generate_caption(url):
    if url in caption_cache:
        return caption_cache[url]
    try:
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        caption = "Image could not be captioned."
    caption_cache[url] = caption
    return caption

def process_markdown(text):
    def replacer(match):
        img_url = match.group(1)
        link_url = match.group(2)
        caption = generate_caption(img_url)
        return f"**Image description**: {caption}\n\n[Link]({link_url})"
    return re.sub(r"\[\!\[.*?\]\((.*?)\)\]\((.*?)\)", replacer, text)

# Paths
input_dir = Path("tds_pages_md")
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

for md_file in input_dir.glob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    cleaned = process_markdown(text)
    (output_dir / md_file.name).write_text(cleaned, encoding="utf-8")

print(f"✅ Cleaned {len(list(output_dir.glob('*.md')))} files with image captions.")
