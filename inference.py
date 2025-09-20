import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------- CONFIG ----------------
model_path = "./results"  # path to your fine-tuned model
images_folder = "images_synthetic"  # folder containing images to OCR
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# ---------------- INFERENCE ----------------
def ocr_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Process all images in the folder
results = {}
for file_name in os.listdir(images_folder):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        file_path = os.path.join(images_folder, file_name)
        text = ocr_image(file_path)
        results[file_name] = text
        print(f"{file_name} -> {text}")

print("\n✅ Batch inference complete!")
