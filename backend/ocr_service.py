import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import re
import io
import fitz
from docx import Document

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set models
PRINTED_MODEL_NAME = "microsoft/trocr-large-printed"
HANDWRITTEN_MODEL_NAME = "microsoft/trocr-large-handwritten"

# Load models globally to avoid reloading on each API call
print(f"Loading TrOCR models to {DEVICE}...")
PROCESSOR = TrOCRProcessor.from_pretrained(PRINTED_MODEL_NAME) # Processor is common
PRINTED_MODEL = VisionEncoderDecoderModel.from_pretrained(PRINTED_MODEL_NAME)
HANDWRITTEN_MODEL = VisionEncoderDecoderModel.from_pretrained(HANDWRITTEN_MODEL_NAME)

PRINTED_MODEL.to(DEVICE)
HANDWRITTEN_MODEL.to(DEVICE)
print("TrOCR models loaded.")

def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Detects and corrects skew in an image.
    """
    # Convert to grayscale and invert for text detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Threshold to get binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find coordinates of all black pixels
    coords = np.column_stack(np.where(thresh > 0))

    # Use minAreaRect to find the minimum bounding rectangle for the text
    angle = cv2.minAreaRect(coords)[-1]

    # The angle returned by minAreaRect is in the range [-90, 0). 
    # Adjust the angle to be in the range [-45, 45) for easier rotation.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get the center of the image for rotation
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhances image quality for better OCR results.
    Applies deskewing, grayscale, denoising, contrast enhancement, and optional upscaling.
    """
    # Temporarily disable deskewing and complex enhancements to debug
    # img_bgr = deskew_image(img_bgr)

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Simple thresholding for initial debugging
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Convert back to BGR for consistency with TrOCR input
    enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Optional: upscale small images to improve OCR
    h, w = enhanced_bgr.shape[:2]
    if w < 1000:
        scale = 1000 / w
        enhanced_bgr = cv2.resize(enhanced_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    return enhanced_bgr

def ocr_line(crop_img: np.ndarray, processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> str:
    """
    Performs OCR on a single line image.
    """
    # Convert OpenCV image to PIL Image for TrOCR
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    
    # Apply additional PIL-based enhancements for line-level OCR
    # crop_pil = ImageOps.autocontrast(crop_pil)
    # crop_pil = ImageEnhance.Sharpness(crop_pil).enhance(2.0)
    # crop_pil = ImageEnhance.Contrast(crop_pil).enhance(2.0)
    # crop_pil = crop_pil.filter(ImageFilter.MedianFilter(size=3))
    
    w_crop, h_crop = crop_pil.size
    if w_crop < 600: # Upscale small lines
        crop_pil = crop_pil.resize((600, int(h_crop*(600/w_crop))), Image.BICUBIC)
    
    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=5, early_stopping=True)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

def run_ocr_on_image(image_bytes: bytes, doc_type: str = "printed") -> str:
    """
    Runs OCR on the entire image, detecting lines and processing them.
    """
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image from provided bytes.")

    # Enhance the entire image first
    img_bgr = enhance_image(img_bgr)

    # Preprocess for line detection: threshold + dilation
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

    # Select model based on doc_type
    current_model = PRINTED_MODEL if doc_type == "printed" else HANDWRITTEN_MODEL

    outputs = []
    for x, y, w, h in line_boxes:
        crop = img_bgr[y:y+h, x:x+w]
        text = ocr_line(crop, PROCESSOR, current_model)
        outputs.append(text)
    
    full_text = "\n".join([t for t in outputs if t])

    # Auto-fallback if output seems empty or garbage (similar to workng.py)
    if len(full_text.strip()) == 0 or set(full_text.strip()) in [{"T"}, {""}]:
        print("Initial model output seems incorrect, switching to other model...")
        current_model = HANDWRITTEN_MODEL if doc_type == "printed" else PRINTED_MODEL
        outputs = []
        for x, y, w, h in line_boxes:
            crop = img_bgr[y:y+h, x:x+w]
            text = ocr_line(crop, PROCESSOR, current_model)
            outputs.append(text)
        full_text = "\n".join([t for t in outputs if t])

    return full_text

def extract_fields(text: str, document_type: str = "general") -> dict:
    """
    Extracts key fields from the OCR text using regex, adaptable for different document types.
    """
    fields = {
        "Name": None,
        "Age": None,
        "Gender": None,
        "Address": None,
        "Email": None,
        "Phone Number": None,
        "DOB": None, # Keep DOB for potential age calculation or ID card context
        "ID Number": None # Keep ID Number for ID card context
    }

    # General patterns applicable to many documents
    # Name: More robust pattern for names, often preceded by labels or at the start of a line
    name_match = re.search(r"(?:Name|Full Name|Applicant Name|Beneficiary)[:\s]*([A-Za-z\s.-]+)", text, re.IGNORECASE)
    if name_match:
        fields['Name'] = name_match.group(1).strip()
    
    # Age: Looks for 'Age:' followed by numbers, or a number followed by 'years old'
    age_match = re.search(r"(?:Age)[:\s]*(\d{1,3})\b|\b(\d{1,3})\s*years\s*old\b", text, re.IGNORECASE)
    if age_match:
        fields['Age'] = age_match.group(1) or age_match.group(2)

    # Gender: Looks for 'Gender:' followed by Male/Female/Other or M/F
    gender_match = re.search(r"(?:Gender)[:\s]*(Male|Female|Other|M|F)\b", text, re.IGNORECASE)
    if gender_match:
        fields['Gender'] = gender_match.group(1).strip()

    # Address: Looks for 'Address:' followed by text, potentially multi-line
    address_match = re.search(r"(?:Address|Residential Address)[:\s]*(.+?)(?:\n|$|Email|Phone|Mobile|DOB|Date of Birth|ID Number|Identity No|Age|Gender)", text, re.IGNORECASE | re.DOTALL)
    if address_match:
        fields['Address'] = address_match.group(1).strip()

    # Email: Standard email regex pattern
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        fields['Email'] = email_match.group(0).strip()

    # Phone Number: Common phone number patterns (e.g., +1 (123) 456-7890, 123-456-7890, 1234567890)
    phone_match = re.search(r"(?:Phone|Mobile|Contact)[:\s]*(\\+?\d{1,3}[-\\s]?\(?\d{3}\)?[-\\s]?\d{3}[-\\s]?\d{4})", text, re.IGNORECASE)
    if phone_match:
        fields['Phone Number'] = phone_match.group(1).strip()

    # ID Card specific patterns (retained from previous version)
    if document_type == "id_card":
        dob_match = re.search(r"(?:DOB|Date of Birth)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
        if dob_match:
            fields['DOB'] = dob_match.group(1).strip()
        
        id_match = re.search(r"(?:ID|ID Number|Identity No)[:\s]*([A-Z0-9]+)", text, re.IGNORECASE)
        if id_match:
            fields['ID Number'] = id_match.group(1).strip()

    return fields


def process_registration_document(file_bytes: bytes, file_type: str, doc_type: str = "printed") -> dict:
    """
    Processes various document types for registration data extraction.
    """
    full_text = ""
    if file_type.startswith("image/"):
        full_text = run_ocr_on_image(file_bytes, doc_type)
    elif file_type == "application/pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            # Fallback to OCR if text extraction fails or is empty
            # Convert first page to image and OCR
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                full_text = run_ocr_on_image(img_bytes, doc_type)
                doc.close()
            except Exception as e_ocr:
                print(f"Error OCRing PDF page: {e_ocr}")
                raise ValueError("Could not extract text from PDF.")
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # .docx
        try:
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                full_text += para.text + "\n"
        except Exception as e:
            print(f"Error processing DOCX: {e}")
            raise ValueError("Could not extract text from DOCX.")
    elif file_type == "text/plain": # .txt
        try:
            full_text = file_bytes.decode('utf-8')
        except Exception as e:
            print(f"Error processing TXT: {e}")
            raise ValueError("Could not decode text file.")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    structured_data = extract_fields(full_text, document_type="general") # Assuming general registration form
    
    return {
        "full_text": full_text,
        "extracted_fields": structured_data
    }
