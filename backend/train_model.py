# train_model.py
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# This is the base model we will fine-tune
BASE_MODEL_NAME = "microsoft/trocr-large-printed"

# This is where the final, trained model will be saved.
# It matches the path in ocr_service.py
FINETUNED_MODEL_PATH = "./trocr-finetuned-model"

# Path to the synthetic dataset you generated
DATASET_PATH = "../synthetic_dataset"
LABELS_FILE = f"{DATASET_PATH}/labels.tsv"
IMAGE_FOLDER = f"{DATASET_PATH}/images"

# --- 1. Load and Prepare Dataset ---
print("Loading dataset...")
try:
    df = pd.read_csv(LABELS_FILE, sep="\t", header=None, names=['file_name', 'text'])
except FileNotFoundError:
    print(f"ERROR: The labels file was not found at '{LABELS_FILE}'.")
    print("Please ensure you have run 'generate_synth_data.py' in the root directory.")
    exit()

# Create the full path to each image
df['file_name'] = df['file_name'].apply(lambda x: f"{IMAGE_FOLDER}/{x}")

# Split data into training and validation sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"Dataset loaded: {len(train_dataset)} training examples, {len(test_dataset)} evaluation examples.")

# --- 2. Initialize Model and Processor ---
print(f"Loading base model '{BASE_MODEL_NAME}' for fine-tuning...")
processor = TrOCRProcessor.from_pretrained(BASE_MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME).to(DEVICE)

# --- 3. Preprocess Data ---
def preprocess_data(examples):
    # Load images
    images = [Image.open(path).convert("RGB") for path in examples['file_name']]
    
    # Process images (resize, normalize)
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    # Process text labels (tokenize)
    labels = processor.tokenizer(examples['text'], padding="max_length", max_length=64, truncation=True).input_ids
    
    # Important: Set padding tokens to be ignored by the loss function
    labels = [[-100 if token == processor.tokenizer.pad_token_id else token for token in label] for label in labels]
    
    return {"pixel_values": pixel_values, "labels": labels}

train_dataset.set_transform(preprocess_data)
test_dataset.set_transform(preprocess_data)

# --- 4. Configure and Run Training ---
print("Configuring training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=FINETUNED_MODEL_PATH,
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=4,  # Lower if you have memory issues
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(), # Use mixed-precision for faster training on CUDA
    num_train_epochs=3,             # Increase for better results, e.g., 5-10
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# Define the metric (Character Error Rate)
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator,
)

print("\n--- Starting Fine-Tuning ---")
trainer.train()
print("\n--- Fine-Tuning Complete ---")

print(f"Saving best model and processor to '{FINETUNED_MODEL_PATH}'...")
trainer.save_model(FINETUNED_MODEL_PATH)
processor.save_pretrained(FINETUNED_MODEL_PATH)
print("Done!")
