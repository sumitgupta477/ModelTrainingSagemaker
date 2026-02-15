import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW


# SageMaker paths
DATA_DIR = "/opt/ml/input/data/training"
MODEL_DIR = os.environ["SM_MODEL_DIR"]

DATA_FILE = os.path.join(DATA_DIR, "data.txt")

MODEL_NAME = "t5-small"   # lightweight text2text model
MAX_LEN = 32
BATCH_SIZE = 2
EPOCHS = 10


# ===============================
# Dataset
# ===============================
class NameDataset(Dataset):
    def __init__(self, file_path, tokenizer):

        self.pairs = []

        with open(file_path, "r") as f:
            for line in f:
                if "=" in line:
                    inp, out = line.split("=")
                    self.pairs.append(
                        (inp.strip(), out.strip())
                    )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        src, tgt = self.pairs[idx]

        x = self.tokenizer(
            src,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        y = self.tokenizer(
            tgt,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": x["input_ids"].squeeze(),
            "attention_mask": x["attention_mask"].squeeze(),
            "labels": y["input_ids"].squeeze()
        }


# ===============================
# Training
# ===============================
def main():

    print("Loading tokenizer/model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = NameDataset(DATA_FILE, tokenizer)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Training started...")

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch in loader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    print("Saving model...")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("Training complete.")


if __name__ == "__main__":
    main()
