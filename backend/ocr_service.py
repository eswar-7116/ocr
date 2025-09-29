import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import re
import io
import fitz
from docx import Document
from thefuzz import fuzz # For fuzzy matching
import json 
import os

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set model paths - CRITICAL CHANGE FOR SPEED: Using 'base' models instead of 'large'
# Note: Ensure this path correctly points to your fine-tuned model folder.
FINETUNED_MODEL_PATH = "./trocr_finetuned_final_model" 
HANDWRITTEN_MODEL_NAME = "microsoft/trocr-small-handwritten" 

# --- Global Model Loading ---
try:
    print(f"Loading TrOCR models to {DEVICE}...")
    
    # Load your fine-tuned model for PRINTED/ID docs
    PROCESSOR = TrOCRProcessor.from_pretrained(FINETUNED_MODEL_PATH)
    PRINTED_MODEL = VisionEncoderDecoderModel.from_pretrained(FINETUNED_MODEL_PATH)
    
    # Load the base model for complex HANDWRITTEN docs
    HANDWRITTEN_MODEL = VisionEncoderDecoderModel.from_pretrained(HANDWRITTEN_MODEL_NAME)

    PRINTED_MODEL.to(DEVICE)
    HANDWRITTEN_MODEL.to(DEVICE)
    
    print("TrOCR models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # Display message if loading fails, but proceed with None models for server health checks
    print("WARNING: Model loading failed. Check your path and files.")
    PROCESSOR, PRINTED_MODEL, HANDWRITTEN_MODEL = None, None, None 
    # NOTE: The API endpoints in main.py must handle the case where models are None!

# --- CORE UTILITY FUNCTIONS (Deskewing, Enhancement, OCR Line, etc.) ---

# NOTE: The body of deskew_image, enhance_image, ocr_line, and run_ocr_on_image
#       is assumed to be correct and remains unchanged from your previous block.

def deskew_image(img: np.ndarray) -> np.ndarray:
    # ... (Your deskew_image code here) ...
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100: return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = deskew_image(img_bgr)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    h, w = enhanced_bgr.shape[:2]
    if w < 1000:
        scale = 1000 / w
        enhanced_bgr = cv2.resize(enhanced_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return enhanced_bgr

def ocr_line(crop_img: np.ndarray, processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> str:
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    w_crop, h_crop = crop_pil.size
    if w_crop < 600 and w_crop > 0:
        crop_pil = crop_pil.resize((600, int(h_crop*(600/w_crop))), Image.BICUBIC)
    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=3, early_stopping=True)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

def run_ocr_on_image(image_bytes: bytes, doc_type: str = "printed") -> str:
    if PRINTED_MODEL is None or HANDWRITTEN_MODEL is None:
        raise Exception("OCR models were not initialized.") # Raise exception if models failed to load
        
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None: raise ValueError("Could not decode image from provided bytes.")
    img_bgr = enhance_image(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 20]
    if not word_boxes: return ""
    word_boxes = sorted(word_boxes, key=lambda b: b[1])
    lines = []
    if word_boxes:
        current_line = [word_boxes[0]]
        for box in word_boxes[1:]:
            prev_box = current_line[-1]
            if (box[1] < prev_box[1] + prev_box[3]): current_line.append(box)
            else: lines.append(current_line); current_line = [box]
        lines.append(current_line)
    current_model = PRINTED_MODEL if doc_type == "printed" else HANDWRITTEN_MODEL
    full_text = ""
    for line_of_boxes in lines:
        line_of_boxes = sorted(line_of_boxes, key=lambda b: b[0])
        x_min = min([b[0] for b in line_of_boxes])
        y_min = min([b[1] for b in line_of_boxes])
        x_max = max([b[0] + b[2] for b in line_of_boxes])
        y_max = max([b[1] + b[3] for b in line_of_boxes])
        line_crop = img_bgr[y_min:y_max, x_min:x_max]
        text = ocr_line(line_crop, PROCESSOR, current_model)
        indentation = " " * (x_min // 20) 
        full_text += indentation + text + "\n"
    return full_text

def extract_fields(text: str, document_type: str = "general") -> dict:
    """Extracts key fields from the OCR text using regex."""
    fields = {
        "Name": None, "Age": None, "Gender": None, "Address": None, 
        "Email": None, "Phone_Number": None, "DOB": None, "ID_Number": None 
    }
    # Name
    name_match = re.search(r"(?:Name|Full Name|Applicant Name|Beneficiary|First name)[:\s]*([A-Za-z\s.'-]+)", text, re.IGNORECASE)
    if name_match: fields['Name'] = name_match.group(1).strip()
    # Age/DOB
    age_match = re.search(r"(?:Age)[:\s]*(\d{1,3})\b|\b(\d{1,3})\s*years\s*old\b", text, re.IGNORECASE)
    if age_match: fields['Age'] = age_match.group(1) or age_match.group(2)
    dob_match = re.search(r"(?:D\.?O\.?B|Date of Birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.IGNORECASE)
    if dob_match: fields['DOB'] = dob_match.group(1).strip()
    # Gender
    gender_match = re.search(r"(?:Gender)[:\s]*(Male|Female|Other|M|F)\b", text, re.IGNORECASE)
    if gender_match: fields['Gender'] = gender_match.group(1).strip()
    # Address: Stops extraction at the next known field label or end of document
    address_match = re.search(r"(?:Address|Residential Address)[:\s]*(.+?)(?=\n(?:Email|Phone|Mobile|DOB|Date of Birth|ID Number|Identity No|Age|Gender)|$)", text, re.IGNORECASE | re.DOTALL)
    if address_match: fields['Address'] = re.sub(r'\s*\n\s*', ' ', address_match.group(1).strip()) 
    # Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match: fields['Email'] = email_match.group(0).strip()
    # Phone Number: Basic pattern for numbers separated by hyphens or spaces
    phone_match = re.search(r"(?:Phone|Mobile|Contact)[:\s]*(\+?\d{1,3}[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4})", text, re.IGNORECASE)
    if phone_match: fields['Phone_Number'] = phone_match.group(1).strip()
    # ID Number (using a broader pattern for alphanumeric IDs)
    id_match = re.search(r"(?:ID|ID Number|Identity No|Identity Number)[:\s]*([A-Z0-9-]{6,})", text, re.IGNORECASE)
    if id_match: fields['ID_Number'] = id_match.group(1).strip()

    return {k: v for k, v in fields.items() if v is not None}

def process_registration_document(file_bytes: bytes, file_type: str, doc_type: str = "printed") -> dict:
    full_text = ""
    if file_type.startswith("image/"):
        full_text = run_ocr_on_image(file_bytes, doc_type)
    elif file_type == "application/pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(doc.page_count):
                full_text += doc.load_page(page_num).get_text()
            doc.close()
        except Exception as e:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                full_text = run_ocr_on_image(img_bytes, doc_type)
                doc.close()
            except Exception as e_ocr:
                raise ValueError(f"Could not extract text from PDF/OCR: {e_ocr}")
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                full_text += para.text + "\n"
        except Exception as e:
            raise ValueError(f"Could not extract text from DOCX: {e}")
    elif file_type == "text/plain": 
        try:
            full_text = file_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Could not decode text file: {e}")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    structured_data = extract_fields(full_text, document_type="general")
    
    return {
        "full_text": full_text,
        "extracted_fields": structured_data
    }

def verify_data(extracted_data: dict, submitted_data: dict) -> dict:
    """
    Compares extracted fields against submitted fields using fuzzy matching 
    and determines a match status and confidence score.
    """
    verification_results = {}
    
    for field, submitted_val in submitted_data.items():
        extracted_val = extracted_data.get(field)
        submitted_val = str(submitted_val).strip().lower()
        
        if extracted_val is None:
            verification_results[field] = {
                "extracted_value": None,
                "submitted_value": submitted_val,
                "confidence_score": 0.0,
                "match_status": "NOT_FOUND"
            }
            continue

        extracted_val = str(extracted_val).strip().lower()
        
        # Use Token Set Ratio: robust comparison metric
        fuzz_score = fuzz.token_set_ratio(submitted_val, extracted_val)
        
        match_threshold = 90
        
        verification_results[field] = {
            "extracted_value": extracted_data.get(field),
            "submitted_value": submitted_data.get(field),
            "confidence_score": round(fuzz_score / 100.0, 3),
            "match_status": "MATCH" if fuzz_score >= match_threshold else "MISMATCH"
        }
        
    return verification_results

# NOTE: The Flask app setup at the end of ocr_service.py has been REMOVED
# as the FastAPI server in main.py will handle execution.