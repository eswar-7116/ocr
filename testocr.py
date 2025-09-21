# exact_paragraph_ocr.py
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

IMAGE_PATH = "sample.png"
MODEL_NAME = "microsoft/trocr-large-printed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

img_bgr = cv2.imread(IMAGE_PATH)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(img_gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
dilated = cv2.dilate(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
line_boxes = [(x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours]
              if w > 50 and h > 15]
line_boxes = sorted(line_boxes, key=lambda b: b[1])

outputs = []
for x, y, w, h in line_boxes:
    crop = img_bgr[y:y+h, x:x+w]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_pil = ImageEnhance.Contrast(crop_pil).enhance(2.0)
    crop_pil = ImageEnhance.Sharpness(crop_pil).enhance(2.0)
    crop_pil = crop_pil.filter(ImageFilter.MedianFilter(size=3))
    w_crop, h_crop = crop_pil.size
    if w_crop < 600:
        crop_pil = crop_pil.resize((600, int(h_crop*(600/w_crop))), Image.BICUBIC)
    pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values, max_length=512, num_beams=5, early_stopping=True)
    outputs.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())

print("\n--- OCR RESULT ---\n")
print("\n".join(outputs))
