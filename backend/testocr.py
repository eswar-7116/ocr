import re
import json
import torch
from PIL import Image, ImageOps, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --------------------------
# Load both models
# --------------------------
print("Loading models... (this may take a while)")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")

printed_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
handwritten_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
printed_model.to(device)
handwritten_model.to(device)

# --------------------------
# Preprocessing function
# --------------------------
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = ImageOps.autocontrast(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    return image

# --------------------------
# Run OCR with fallback
# --------------------------
def run_ocr(image, doc_type):
    if doc_type == "printed":
        model = printed_model
    else:
        model = handwritten_model

    # Process with first model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Fallback if text is too small (failed OCR)
    if len(text.strip()) < 5:
        print("⚠️ Switching model because OCR failed...")
        model = handwritten_model if doc_type == "printed" else printed_model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text

# --------------------------
# Extract fields
# --------------------------
def extract_fields(text):
    fields = {
        "Name": None,
        "DOB": None,
        "ID Number": None,
        "Address": None
    }

    # Regex patterns for fields
    dob_pattern = re.compile(r"(\d{2}[/-]\d{2}[/-]\d{4})")
    id_pattern = re.compile(r"\b([A-Z0-9]{6,})\b")

    for line in text.split("\n"):
        line = line.strip()

        if "name" in line.lower():
            fields["Name"] = line.split(":")[-1].strip()

        if "dob" in line.lower() or "date of birth" in line.lower():
            dob_match = dob_pattern.search(line)
            if dob_match:
                fields["DOB"] = dob_match.group(1)

        if "address" in line.lower():
            fields["Address"] = line.split(":")[-1].strip()

        if "id" in line.lower() or "number" in line.lower():
            id_match = id_pattern.search(line)
            if id_match:
                fields["ID Number"] = id_match.group(1)

    return fields

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    img_path = input("Enter path of document image: ").strip()
    doc_type = input("Is this document handwritten or printed? (handwritten/printed): ").strip().lower()

    image = preprocess_image(img_path)
    extracted_text = run_ocr(image, doc_type)

    print("\n----- OCR RAW OUTPUT -----")
    print(extracted_text)

    fields = extract_fields(extracted_text)

    print("\n----- STRUCTURED OUTPUT -----")
    print(json.dumps(fields, indent=4))
