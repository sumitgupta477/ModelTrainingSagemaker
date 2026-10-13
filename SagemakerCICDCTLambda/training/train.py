"""Fine-tune a small Hugging Face seq2seq model (T5) on SageMaker.

Training data: every *.txt file in the "training" channel (s3://<bucket>/input/),
one example per line in the form  `input text = target text`.

Supports managed spot training: a checkpoint is written to /opt/ml/checkpoints
after every epoch (SageMaker syncs it to S3), and training resumes from the
latest checkpoint if the spot instance is interrupted.
"""

import glob
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# SageMaker paths
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CHECKPOINT_DIR = "/opt/ml/checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")

MODEL_NAME = os.environ.get("BASE_MODEL", "t5-small")  # lightweight text2text model
MAX_LEN = 32
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 5e-5


class PairDataset(Dataset):
    """Reads `input = target` pairs from every .txt file in the data directory."""

    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        self.pairs = []
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        # Split on the first "=" only, so targets may contain "=".
                        src, tgt = line.split("=", 1)
                        if src.strip() and tgt.strip():
                            self.pairs.append((src.strip(), tgt.strip()))
        print(f"Loaded {len(self.pairs)} training pairs from {len(files)} file(s)")
        if not self.pairs:
            raise ValueError(f"No 'input = target' lines found in {data_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        x = self.tokenizer(src, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
        y = self.tokenizer(tgt, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
        labels = y["input_ids"].squeeze()
        # Ignore padding tokens in the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": x["input_ids"].squeeze(),
            "attention_mask": x["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    state = torch.load(CHECKPOINT_FILE, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"Resumed from checkpoint after epoch {state['epoch']}")
    return state["epoch"]


def save_checkpoint(model, optimizer, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp = CHECKPOINT_FILE + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch}, tmp)
    os.replace(tmp, CHECKPOINT_FILE)


def main():
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = PairDataset(DATA_DIR, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoint(model, optimizer)

    print("Training started...")
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        # SageMaker parses this line into the training job's metrics.
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")
        save_checkpoint(model, optimizer, epoch + 1)

    print("Saving model...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Training complete.")


if __name__ == "__main__":
    main()
