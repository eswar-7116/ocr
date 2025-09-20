from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import json
import os
import random

# ---------------- CONFIG ----------------
output_folder = "images_synthetic"
os.makedirs(output_folder, exist_ok=True)

dataset_json = []

# Fonts (TTF)
regular_font = r"D:\Project\fonts\Geist-Regular.ttf"
mono_font = r"D:\Project\fonts\Geist-Mono-Regular.ttf"
base_font_size = 32

# Website colors
bg_color = (10, 10, 10)
text_color = (237, 237, 237)

# Image size
image_width = 1024
image_height = 128

# Texts to generate
texts = [
    "I'm Eswar Dudi, a software developer passionate about backend systems, AI, and",
    "web development. I enjoy building awesome projects and learning deep about tech.",
    "About me",
    "Some of my top projects include NexusChat (a peer-to-peer chat app with JWT auth",
    "and real-time communication), Guntainer (a container runtime in Go that isolates",
    "processes), and CalGist (an AI-powered calendar summarizer). You can find more in"
]

# Number of variations per text
variations = 10

# ---------------- GENERATE IMAGES ----------------
img_counter = 1
for text in texts:
    for v in range(variations):
        # Randomize font size slightly
        font_size = base_font_size + random.randint(-3, 3)
        font = ImageFont.truetype(regular_font, font_size)

        # Randomize background brightness slightly
        brightness_factor = random.uniform(0.9, 1.1)
        bg_color_var = tuple(min(255, max(0, int(c * brightness_factor))) for c in bg_color)

        # Create blank image
        img = Image.new("RGB", (image_width, image_height), color=bg_color_var)
        draw = ImageDraw.Draw(img)

        # Wrap text
        lines = []
        words = text.split()
        line = ""
        for word in words:
            if draw.textlength(line + " " + word, font=font) < image_width - 20:
                line += " " + word if line else word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        # Draw text with slight position variations
        y_text = 10 + random.randint(-5, 5)
        for line in lines:
            x_text = 10 + random.randint(-5, 5)
            draw.text((x_text, y_text), line, font=font, fill=text_color)
            y_text += font_size + random.randint(0, 5)

        # Save image
        file_name = f"img_synth_{img_counter}.png"
        file_path = os.path.join(output_folder, file_name)
        img.save(file_path)

        # Append to JSON
        dataset_json.append({
            "file": f"{output_folder}/{file_name}",
            "text": text
        })

        img_counter += 1

# Save dataset JSON
with open("dataset_synthetic.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ Generated {len(dataset_json)} synthetic images with variations!")
