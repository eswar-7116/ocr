import os
from PIL import Image, ImageDraw, ImageFont
import random
import csv
from itertools import chain

# --- Configuration ---
OUTPUT_DIR = "synthetic_dataset_v2" # Use a new directory to keep organized
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
NUM_IMAGES_PER_STYLE = 250 # Increase to 1000s for a full fine-tuning run
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 100

# --- Setup ---
FONT_DIR = "fonts"
if not os.path.exists(FONT_DIR) or not os.listdir(FONT_DIR):
    # This check is good, ensure you have both handwritten and printed fonts!
    raise FileNotFoundError("Please create a 'fonts' folder and add .ttf font files (handwritten & printed) to it.")
FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]

# Use separate lists for different styles of text/fields
HANDWRITTEN_TEXT = [
    "Full Name: Jane Doe", "Address: 123 Elm Street", "Date: 01/01/2024", 
    "Total: $125.50", "Signature Here", "Account: 9988-7766",
]
ID_CARD_TEXT = [
    "ID NUMBER: GBR-12345-XYZ", "NAME: JOHNATHAN DOE", "DOB: 1985-08-14",
    "GENDER: MALE", "EXPIRY: 2030-01-01", "CITY: LONDON",
]
SAMPLE_TEXT = HANDWRITTEN_TEXT + ID_CARD_TEXT


# --- Helper Functions ---
def generate_id_card_style():
    """Generates strictly white background and black/dark gray text (ID style)."""
    bg_color = (255, 255, 255)  # White
    text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)) # Very dark gray/black
    return bg_color, text_color

def generate_random_contrast_style():
    """Generates random, high-contrast colors (Original logic)."""
    bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    fg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Ensure a minimum brightness difference
    while abs(sum(bg_color) - sum(fg_color)) < 200:
        fg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return bg_color, fg_color

def generate_image(text, style="random", index=0):
    """Generates a single image with text and returns filename, label."""
    if style == "id_card":
        bg_color, text_color = generate_id_card_style()
        texts = ID_CARD_TEXT
    else: # random or handwritten
        bg_color, text_color = generate_random_contrast_style()
        texts = HANDWRITTEN_TEXT

    font_path = random.choice(FONTS)
    font_size = random.randint(28, 42)
    font = ImageFont.truetype(font_path, font_size)
    
    # Create the image
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Position and draw the text
    text_width, text_height = draw.textbbox((0, 0), text, font=font, anchor="mm")[2:]
    x = IMAGE_WIDTH / 2
    y = IMAGE_HEIGHT / 2
    
    # Add minor random offset for robustness
    x += random.randint(-10, 10)
    y += random.randint(-5, 5)

    draw.text((x, y), text, font=font, fill=text_color, anchor="mm")

    # Save the image and its label
    image_filename = f"{style}_{index:04d}.png"
    image.save(os.path.join(IMAGE_DIR, image_filename))
    return image_filename, text

# --- Main Generation Loop ---
if __name__ == "__main__":
    os.makedirs(IMAGE_DIR, exist_ok=True)
    labels = []

    print(f"Generating {NUM_IMAGES_PER_STYLE} ID Card style images...")
    # Generate ID Card Style (Targeted Training)
    for i in range(NUM_IMAGES_PER_STYLE):
        text = random.choice(ID_CARD_TEXT)
        labels.append(generate_image(text, style="id_card", index=i))

    print(f"Generating {NUM_IMAGES_PER_STYLE} Random/Handwritten style images...")
    # Generate Random/Handwritten Style (General Robustness)
    for i in range(NUM_IMAGES_PER_STYLE):
        text = random.choice(HANDWRITTEN_TEXT)
        labels.append(generate_image(text, style="random", index=i))

    # Save all labels to a single file
    labels_filepath = os.path.join(OUTPUT_DIR, "labels.tsv")
    with open(labels_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_name", "text"]) # Add header for clarity
        writer.writerows(labels)

    print(f"Done! Total {len(labels)} synthetic images created in '{OUTPUT_DIR}'")
    print(f"Labels file created at '{labels_filepath}'")



# # generate_synth_data.py
# import os
# from PIL import Image, ImageDraw, ImageFont
# import random
# import csv

# # --- Configuration ---
# OUTPUT_DIR = "synthetic_dataset"
# IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
# NUM_IMAGES = 100  # Start with 100, increase to 1000s for real training
# IMAGE_WIDTH = 800
# IMAGE_HEIGHT = 100

# # --- Setup ---
# # 1. Create a folder named 'fonts' and place .ttf font files inside.
# #    Download handwritten fonts from https://fonts.google.com/?category=Handwriting
# FONT_DIR = "fonts"
# if not os.path.exists(FONT_DIR) or not os.listdir(FONT_DIR):
#     raise FileNotFoundError("Please create a 'fonts' folder and add .ttf font files to it.")
# FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]

# # 2. Create a list of sample text to generate.
# SAMPLE_TEXT = [
#     "Full Name: Johnathan Doe", "ID Number: GBR-12345-XYZ", "Date of Birth: 14/08/1985",
#     "Address: 123 Innovation Drive, Tech City", "Gender: Male", "Phone: (555) 867-5309",
#     "Applicant Name: Priya Sharma", "Identity No: 9876-5432-10", "DOB: 03-MAR-1992",
#     "Residential Address: Apt 4B, Silicon Valley", "Email: test.user@email.com",
# ]

# # --- Helper Functions ---
# def get_random_color():
#     """Generates a random RGB color tuple."""
#     return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# def get_contrasting_colors():
#     """Ensures text is readable against the background."""
#     bg_color = get_random_color()
#     fg_color = get_random_color()
#     # Ensure a minimum brightness difference (luminance)
#     while abs(sum(bg_color) - sum(fg_color)) < 200:
#         fg_color = get_random_color()
#     return bg_color, fg_color

# # --- Main Generation Loop ---
# if __name__ == "__main__":
#     os.makedirs(IMAGE_DIR, exist_ok=True)
#     labels = []

#     print(f"Generating {NUM_IMAGES} synthetic images...")

#     for i in range(NUM_IMAGES):
#         # 1. Get random properties for the image
#         text = random.choice(SAMPLE_TEXT)
#         font_path = random.choice(FONTS)
#         font_size = random.randint(28, 42)
#         font = ImageFont.truetype(font_path, font_size)
#         bg_color, text_color = get_contrasting_colors()

#         # 2. Create the image
#         image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=bg_color)
#         draw = ImageDraw.Draw(image)

#         # 3. Position and draw the text
#         text_width, text_height = draw.textbbox((0,0), text, font=font)[2:]
#         x = (IMAGE_WIDTH - text_width) / 2
#         y = (IMAGE_HEIGHT - text_height) / 2
#         draw.text((x, y), text, font=font, fill=text_color)

#         # 4. Save the image and its label
#         image_filename = f"synth_{i:04d}.png"
#         image.save(os.path.join(IMAGE_DIR, image_filename))
#         labels.append([image_filename, text])

#     # 5. Save all labels to a single file (required for training)
#     labels_filepath = os.path.join(OUTPUT_DIR, "labels.tsv")
#     with open(labels_filepath, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f, delimiter="\t")
#         for row in labels:
#             writer.writerow(row)

#     print(f"Done! Synthetic dataset created in '{OUTPUT_DIR}'")
#     print(f"Labels file created at '{labels_filepath}'")

