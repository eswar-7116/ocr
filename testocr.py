from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")  # keep same processor
model = VisionEncoderDecoderModel.from_pretrained("results").to(device)     # load your trained model


# --- Preprocessing function ---
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # convert to grayscale
    image = ImageOps.invert(image)               # invert colors (white bg, black text)
    image = ImageEnhance.Contrast(image).enhance(2.0)  # boost contrast
    image = ImageEnhance.Sharpness(image).enhance(2.0) # boost sharpness
    image = image.resize((image.width * 2, image.height * 2))  # enlarge
    return image.convert("RGB")  # model expects RGB

# Load and preprocess
image_path = "sample.png"
image = preprocess_image(image_path)

# Convert to model input
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# OCR
generated_ids = model.generate(pixel_values, max_length=128)
recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("OCR Output:", recognized_text)
