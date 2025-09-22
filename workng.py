# smart_document_ocr.py
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ask user for document type
doc_type = input("Enter document type (handwritten / printed): ").strip().lower()

# Set models
printed_model_name = "microsoft/trocr-large-printed"
handwritten_model_name = "microsoft/trocr-large-handwritten"

def load_model(model_name):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(DEVICE)
    return processor, model

# Try loading chosen model first
if doc_type == "handwritten":
    processor, model = load_model(handwritten_model_name)
else:
    processor, model = load_model(printed_model_name)

# Load image
IMAGE_PATH = input("Enter image path: ").strip()
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# -------- IMAGE ENHANCEMENT / DENOISING / RESOLUTION --------
def enhance_image(img_bgr):
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Denoise (fast Non-local Means Denoising)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Optional: upscale small images to improve OCR
    h, w = enhanced.shape
    if w < 800:
        scale = 800 / w
        enhanced = cv2.resize(enhanced, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

img_bgr = enhance_image(img_bgr)

# Preprocess: threshold + dilation
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(img_gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find line contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
line_boxes = [(x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours] if w > 30 and h > 10]
line_boxes = sorted(line_boxes, key=lambda b: b[1])

def ocr_line(crop_img, processor, model):
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    crop_pil = ImageEnhance.Contrast(crop_pil).enhance(2.0)
    crop_pil = ImageEnhance.Sharpness(crop_pil).enhance(2.0)
    crop_pil = crop_pil.filter(ImageFilter.MedianFilter(size=3))
    w_crop, h_crop = crop_pil.size
    if w_crop < 600:
        crop_pil = crop_pil.resize((600, int(h_crop*(600/w_crop))), Image.BICUBIC)
    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=5, early_stopping=True)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

def run_ocr(processor, model):
    outputs = []
    for x, y, w, h in line_boxes:
        crop = img_bgr[y:y+h, x:x+w]
        text = ocr_line(crop, processor, model)
        outputs.append(text)
    return "\n".join([t for t in outputs if t])

# -------- EXTRACT KEY FIELDS FROM ID CARD --------
def extract_fields(text):
    fields = {}
    
    # Name
    name_match = re.search(r"(?:Name|Full Name)[:\s]*([A-Za-z\s]+)", text, re.IGNORECASE)
    if name_match:
        fields['Name'] = name_match.group(1).strip()
    
    # DOB
    dob_match = re.search(r"(?:DOB|Date of Birth)[:\s]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", text, re.IGNORECASE)
    if dob_match:
        fields['DOB'] = dob_match.group(1).strip()
    
    # ID Number
    id_match = re.search(r"(?:ID|ID Number|Identity No)[:\s]*([A-Z0-9]+)", text, re.IGNORECASE)
    if id_match:
        fields['ID Number'] = id_match.group(1).strip()
    
    # Address
    address_match = re.search(r"(?:Address)[:\s]*(.+)", text, re.IGNORECASE)
    if address_match:
        fields['Address'] = address_match.group(1).strip()
    
    return fields

# Run OCR with chosen model
full_text = run_ocr(processor, model)

# Auto-fallback if output seems empty or garbage
if len(full_text.strip()) == 0 or set(full_text.strip()) in [{"T"}, {""}]:
    print("Initial model output seems incorrect, switching to other model...")
    if doc_type == "handwritten":
        processor, model = load_model(printed_model_name)
    else:
        processor, model = load_model(handwritten_model_name)
    full_text = run_ocr(processor, model)

# Extract structured fields
structured_data = extract_fields(full_text)

print("\n--- OCR RESULT ---\n")
print(full_text)

print("\n--- EXTRACTED FIELDS ---\n")
if structured_data:
    for k, v in structured_data.items():
        print(f"{k}: {v}")
else:
    print("No key fields detected.")