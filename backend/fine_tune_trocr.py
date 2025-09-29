import pandas as pd
from datasets import Dataset
import evaluate 
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import os
import random
import warnings

# Suppress Hugging Face warnings during training, as they can be verbose
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = "synthetic_dataset_v2" # Match the output directory from generate_synth_data.py
LABELS_FILE = os.path.join(DATA_DIR, "labels.tsv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_OUTPUT_DIR = "trocr_finetuned_model"
TRAINING_MODEL = "microsoft/trocr-base-handwritten" # Good starting point for varied documents

# Check for labels file existence
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}. Please run generate_synth_data.py first.")

# --- 1. Custom Dataset Class (Handles Images and Augmentation) ---

class OCRDataset(TorchDataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We assume the TSV columns are 'file_name' and 'text'
        try:
            file_name, text = self.df.iloc[idx]['file_name'], self.df.iloc[idx]['text']
        except KeyError:
            # Fallback for old pandas versions or weirdly formatted TSVs
            file_name, text = self.df.iloc[idx].iloc[0], self.df.iloc[idx].iloc[1]
            
        image_path = os.path.join(IMAGES_DIR, file_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None # Skip this sample

        # --- Data Augmentation: CRITICAL for robustness ---
        # 1. Random Color Inversion (Fixes black/white ID contrast issue)
        if random.random() < 0.3: # 30% chance of inversion
            image = ImageOps.invert(image)

        # Prepare image (resize + normalize)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Prepare text (tokenization)
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids

        # Important: make sure PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # Squeeze pixel values to remove the batch dimension added by the processor
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def collate_fn(batch):
    """Handles None values (skipped samples) and combines samples into a batch."""
    # Filter out None values (skipped samples)
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    # Use default PyTorch collation for the rest
    return torch.utils.data.dataloader.default_collate(batch)


# --- 2. Load Data and Split ---

# Load the dataset from the generated TSV file
df = pd.read_csv(LABELS_FILE, sep='\t')

# Ensure columns are named correctly, as generated_synth_data.py uses headers
if 'file_name' not in df.columns:
    df.columns = ['file_name', 'text']
    
if len(df) == 0:
    raise ValueError("Loaded dataset is empty. Check generate_synth_data.py output.")

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")

# --- 3. Initialize Model and Processor ---

processor = TrOCRProcessor.from_pretrained(TRAINING_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TRAINING_MODEL)

# *************************************************************************
# FIX 1: Set the decoder start token ID
# This resolves the "Make sure to set the decoder_start_token_id" error.
model.config.decoder_start_token_id = processor.tokenizer.pad_token_id 
# *************************************************************************

# Configure the model for generation (decoding)
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4 # Use beam search for better predictions

# Create Dataset objects
train_dataset = OCRDataset(train_df, processor)
eval_dataset = OCRDataset(val_df, processor)


# --- 4. Define Metrics ---
cer_metric = evaluate.load("cer") 

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Replace -100 (ignored tokens) with the padding token ID before decoding labels
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# --- 5. Define Training Arguments and Trainer ---

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    eval_strategy="steps", # FIX 2: Renamed from evaluation_strategy to eval_strategy
    do_train=True,
    do_eval=True,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    num_train_epochs=5, # Fine-tuning usually needs fewer epochs
    fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
    learning_rate=5e-5, # Low learning rate for fine-tuning
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn, # Use the custom collator
)


# --- 6. Train and Save ---

print("\nStarting fine-tuning...")

try:
    trainer.train()
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    print("If the error is related to CUDA/GPU, try setting fp16=False in Seq2SeqTrainingArguments.")
    # Exit gracefully if training fails
    exit(1)


# Save the final model and processor to be used by ocr_service.py
final_model_path = os.path.join(MODEL_OUTPUT_DIR, "final")
trainer.save_model(final_model_path)
processor.save_pretrained(final_model_path)
print(f"\nFine-tuning complete. Model saved to '{final_model_path}'")