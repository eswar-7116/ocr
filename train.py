import os
import json
from PIL import Image
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from torch.utils.data import Dataset

# ---------------- CONFIG ----------------
dataset_json_path = "dataset_synthetic.json"
model_name = "microsoft/trocr-base-printed"
output_dir = "./results"
max_target_length = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class OCRDataset(Dataset):
    def __init__(self, dataset_json, processor):
        with open(dataset_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["file"]).convert("RGB")
        text = item["text"]

        # Preprocess image and text
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, truncation=True, max_length=max_target_length, padding="max_length"
        ).input_ids
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}

# ---------------- MAIN ----------------
def main():
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    # Ensure decoder_start_token_id is set (avoids previous training error)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

    # Load dataset
    dataset = OCRDataset(dataset_json_path, processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        data_collator=lambda data: {
            "pixel_values": torch.stack([f["pixel_values"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
        },
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Training complete! Model saved in {output_dir}")

if __name__ == "__main__":
    main()
