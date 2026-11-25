import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
import logging
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import List
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
import random
from itertools import product
from transformers import BertConfig, BertModel


logger = logging.getLogger(__name__)


def configure_logging(log_fname):
    """Send INFO and higher log messages to both the console and a file."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_parent = os.path.dirname(os.path.abspath(log_fname))
    os.makedirs(log_parent, exist_ok=True)
    file_handler = logging.FileHandler(log_fname, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

SEED = 42
MAX_POSITION_EMBEDDINGS = 514


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



class DNATokenizer:
    def __init__(self, k=3):
        self.k = k

        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        kmers = ["".join(p) for p in product("ACGT", repeat=k)]

        self.vocab = self.special_tokens + kmers

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.mask_token_id = self.token_to_id[self.mask_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, sequence):
        sequence = sequence.upper()
        return [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token[idx] for idx in ids]

    def encode(self, sequence, max_length=MAX_POSITION_EMBEDDINGS, pad_to_length=None):
        tokens = self.tokenize(sequence)
        ids = self.convert_tokens_to_ids(tokens)

        ids = [self.cls_token_id] + ids + [self.sep_token_id]

        if len(ids) > max_length:
            raise ValueError(f"Sequence is too long ({len(ids)} > {max_length})")

        if pad_to_length is None:
            pad_to_length = max_length
        if len(ids) > pad_to_length:
            raise ValueError(f"Padding length is too short ({pad_to_length} < {len(ids)})")

        attention_mask = [1] * len(ids)
        padding = pad_to_length - len(ids)
        ids.extend([self.pad_token_id] * padding)
        attention_mask.extend([0] * padding)

        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        return tokens

    def __len__(self):
        return self.vocab_size


class DNANonOverlappingTokenizer(DNATokenizer):
    def tokenize(self, sequence):
        sequence = sequence.upper()
        return [sequence[i : i + self.k] for i in range(0, len(sequence) - self.k + 1, self.k)]



class DNABertExtractor:

    def __init__(self, name_model, device="cuda"):
        self.device = device
        self.name_model = name_model

        self.model_path = os.getenv("DNABERT_MODEL_PATH")

        self.mode = self._infer_mode()

        self._init_tokenizer()

        self.config = self._build_config()
        self.model = BertModel(self.config)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            state_dict = {
                k.replace("bert.", ""): v for k, v in state_dict.items() if not k.startswith("cls.")
            }
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = self.model.to(device)
        self.model.eval()

    def _infer_mode(self):
        mode_env = os.getenv("DNABERT_MODE")
        if mode_env:
            return mode_env

        model_name = os.path.basename(self.model_path).lower()

        if "nucleotide" in model_name:
            return "nucleotide"
        elif "3mer_no" in model_name:
            return "3mer_no"
        elif "6mer_no" in model_name:
            return "6mer_no"
        elif "3mer" in model_name:
            return "3mer"
        elif "6mer" in model_name:
            return "6mer"
        elif "bpe" in model_name:
            return "bpe"
        else:
            raise ValueError(
                f"Could not infer mode from model path: {self.model_path}. "
                "Set DNABERT_MODE environment variable explicitly."
            )

    def _init_tokenizer(self):
        k_map = {
            "nucleotide": 1,
            "3mer": 3,
            "6mer": 6,
            "3mer_no": 3,
            "6mer_no": 6,
        }

        if self.mode in k_map:
            k = k_map[self.mode]
            if self.mode.endswith("_no"):
                self.tokenizer = DNANonOverlappingTokenizer(k=k)
            else:
                self.tokenizer = DNATokenizer(k=k)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_config(self):
        return BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            output_hidden_states=False,
        )

    def _max_sequence_length(self):
        token_budget = MAX_POSITION_EMBEDDINGS - 2
        if isinstance(self.tokenizer, DNANonOverlappingTokenizer):
            return token_budget * self.tokenizer.k
        return token_budget + self.tokenizer.k - 1

    def _chunk_sequence(self, sequence):
        max_length = self._max_sequence_length()
        return [
            sequence[start : start + max_length]
            for start in range(0, len(sequence), max_length)
        ] or [""]

    def extract_embeddings(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        chunked_sequences = [self._chunk_sequence(seq or "") for seq in sequences]
        chunks = [chunk for sequence_chunks in chunked_sequences for chunk in sequence_chunks]
        hidden_size = self.config.hidden_size
        chunk_embeddings = np.empty((len(chunks), hidden_size), dtype=np.float32)
        current_batch_size = max(1, batch_size)

        with torch.inference_mode():
            for i in range(0, len(chunks), current_batch_size):
                batch_sequences = chunks[i : i + current_batch_size]

                batch_length = max(
                    len(self.tokenizer.tokenize(seq)) + 2 for seq in batch_sequences
                )
                encoded = [
                    self.tokenizer.encode(seq, pad_to_length=batch_length)
                    for seq in batch_sequences
                ]

                input_ids = torch.tensor(
                    [enc["input_ids"] for enc in encoded],
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.tensor(
                    [enc["attention_mask"] for enc in encoded],
                    dtype=torch.long,
                    device=self.device,
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            # TODO: проверить нужно ли переносить на cpu
                chunk_embeddings[i : i + len(batch_sequences)] = cls_embeddings.cpu().numpy()
                del outputs, input_ids, attention_mask, encoded, cls_embeddings


        embeddings = []
        offset = 0
        for sequence_chunks in chunked_sequences:
            count = len(sequence_chunks)
            embeddings.append(chunk_embeddings[offset : offset + count].mean(axis=0))
            offset += count

        return np.vstack(embeddings)


def load_data_multilabel(
    train_csv_path,
    test_csv_path,
    train_labels_path,
    test_labels_path,
    sequence_column,
):
    """Load sequences from CSV files and their multi-hot labels from NPY files."""
    train_sequences = pd.read_csv(train_csv_path)[sequence_column].astype(str).tolist()
    test_sequences = pd.read_csv(test_csv_path)[sequence_column].astype(str).tolist()
    train_labels = np.load(train_labels_path, allow_pickle=True).astype(np.float32)
    test_labels = np.load(test_labels_path, allow_pickle=True).astype(np.float32)

    if train_labels.shape[0] != len(train_sequences) or test_labels.shape[0] != len(test_sequences):
        raise ValueError("Each label array must have one row per sequence")
    if train_labels.shape[1] != test_labels.shape[1]:
        raise ValueError("Train and test label arrays must have the same number of classes")

    return train_sequences, train_labels, test_sequences, test_labels


class MultiLabelDNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=514):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        max_tokens = self.max_length - 2 # CLS and SEP
        self.max_chars_per_chunk = max_tokens + self.tokenizer.k - 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        chunks = [seq[start : start + self.max_chars_per_chunk]
                  for start in range(0, max(1, len(seq)), self.max_chars_per_chunk)]

        input_ids_list = []
        attention_mask_list = []

        for chunk in chunks:
            encoded = self.tokenizer.encode(chunk, max_length=self.max_length, pad_to_length=self.max_length)
            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float32)
        }

def collate_chunks_multilabel(batch):
    max_chunks = max(item["input_ids"].shape[0] for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    chunk_masks = []

    for item in batch:
        num_chunks = item["input_ids"].shape[0]
        pad_len = max_chunks - num_chunks

        if pad_len > 0:
            pad_ids = torch.zeros((pad_len, item["input_ids"].shape[1]), dtype=torch.long)
            pad_mask = torch.zeros((pad_len, item["attention_mask"].shape[1]), dtype=torch.long)

            ids = torch.cat([item["input_ids"], pad_ids], dim=0)
            mask = torch.cat([item["attention_mask"], pad_mask], dim=0)
        else:
            ids = item["input_ids"]
            mask = item["attention_mask"]

        chunk_mask = torch.cat([torch.ones(num_chunks), torch.zeros(pad_len)])

        batch_input_ids.append(ids)
        batch_attention_mask.append(mask)
        batch_labels.append(item["labels"])
        chunk_masks.append(chunk_mask)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
        "chunk_masks": torch.stack(chunk_masks)
    }

class LoRADNAMultiLabelClassifier(nn.Module):
    def __init__(self, base_extractor, num_classes, r=8):
        super().__init__()
        self.bert = base_extractor.model

        lora_config = LoraConfig(
            r=r,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)

        self.classifier = nn.Sequential(
            nn.Linear(base_extractor.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, chunk_masks):
        batch_size, num_chunks, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = cls_output.view(batch_size, num_chunks, -1)

        chunk_masks = chunk_masks.unsqueeze(-1).to(cls_output.device)
        sum_embeddings = (cls_output * chunk_masks).sum(dim=1)
        valid_chunks = chunk_masks.sum(dim=1)

        avg_embeddings = sum_embeddings / valid_chunks
        logits = self.classifier(avg_embeddings)
        return logits

def train_lora_multilabel(args):
    set_seed(SEED)

    train_sequences, train_labels, test_sequences, test_labels = load_data_multilabel(
        args.train_csv,
        args.test_csv,
        args.train_labels_npy,
        args.test_labels_npy,
        args.sequence_column,
    )
    num_classes = train_labels.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["DNABERT_MODEL_PATH"] = args.model_weights
    os.environ["DNABERT_MODE"] = args.mode
    base_extractor_template = DNABertExtractor(name_model=args.mode + "_model", device=device)

    train_dataset = MultiLabelDNASequenceDataset(
        train_sequences, train_labels, base_extractor_template.tokenizer
    )
    val_dataset = MultiLabelDNASequenceDataset(
        test_sequences, test_labels, base_extractor_template.tokenizer
    )
    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_chunks_multilabel, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_chunks_multilabel)

    lora_rs = [8, 16]
    lora_lrs = [1e-5, 3e-5, 5e-5, 7e-5, 10e-5]
    head_lr = 1e-3
    epochs = args.epochs
    patience = args.patience

    best_overall_val_loss = float('inf')
    best_lr = None
    best_r = None

    for lora_r in lora_rs:
        for lora_lr in lora_lrs:
            logger.info(f"\n--- Starting Multi-label trial with LoRA R: {lora_r}, LoRA LR: {lora_lr}, Head LR: {head_lr} ---")

            base_extractor = DNABertExtractor(name_model=args.mode + "_model", device=device)
            model = LoRADNAMultiLabelClassifier(base_extractor, num_classes=num_classes, r=lora_r).to(device)

            if lora_lr == lora_lrs[0] and lora_r == lora_rs[0]:
                 model.bert.print_trainable_parameters()

            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.bert.named_parameters() if p.requires_grad], "lr": lora_lr},
                {"params": [p for n, p in model.classifier.named_parameters() if p.requires_grad], "lr": head_lr},
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
            criterion = nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            patience_counter = 0

            total_steps = epochs * len(train_loader)
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            for epoch in range(epochs):
                model.train()
                total_loss = 0

                for i, batch in enumerate(train_loader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    chunk_masks = batch["chunk_masks"].to(device)
                    batch_labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    logits = model(input_ids, attention_mask, chunk_masks)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

                model.eval()
                val_loss = 0
                all_probs = []
                all_labels = []
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        chunk_masks = batch["chunk_masks"].to(device)
                        batch_labels = batch["labels"].to(device)

                        logits = model(input_ids, attention_mask, chunk_masks)
                        loss = criterion(logits, batch_labels)
                        val_loss += loss.item()

                        probs = torch.sigmoid(logits)

                        all_probs.append(probs.cpu().numpy())
                        all_labels.append(batch_labels.cpu().numpy())

                all_probs = np.vstack(all_probs)
                all_labels = np.vstack(all_labels)

                avg_val_loss = val_loss / len(val_loader)

                try:
                    val_prauc = average_precision_score(all_labels, all_probs, average="macro")
                    val_roc_auc = roc_auc_score(all_labels, all_probs, average="macro")
                except ValueError:
                    val_prauc = 0.0
                    val_roc_auc = 0.0

                logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PRAUC: {val_prauc:.4f} | Val ROC AUC: {val_roc_auc:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    if best_val_loss < best_overall_val_loss:
                        best_overall_val_loss = best_val_loss
                        best_lr = lora_lr
                        best_r = lora_r
                        model.bert.save_pretrained(f"lora_dnabert_{args.mode}_multilabel_best")
                        torch.save(model.classifier.state_dict(), f"lora_classifier_{args.mode}_multilabel_head_best.pt")
                        logger.info(f"New best overall model found! Saved to 'lora_dnabert_{args.mode}_multilabel_best/' and 'lora_classifier_{args.mode}_multilabel_head_best.pt'")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered for R {lora_r}, LR {lora_lr} after {epoch+1} epochs.")
                        break

    logger.info(f"\nHyperparameter search finished. Best LoRA R: {best_r}, LR: {best_lr} with Val Loss: {best_overall_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label LoRA fine-tuning for DNABERT")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Prepared training CSV containing sequences")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Prepared test CSV containing sequences")
    parser.add_argument("--train_labels_npy", type=str, default="train_labels.npy", help="Training multi-hot labels NPY file")
    parser.add_argument("--test_labels_npy", type=str, default="test_labels.npy", help="Test multi-hot labels NPY file")
    parser.add_argument("--sequence_column", type=str, default="sequence", help="Name of the sequence column in both CSV files")
    parser.add_argument("--model_weights", type=str, default="dnabert_6mer_25ep_cad_best.pt", help="Path to base model weights")
    parser.add_argument("--mode", type=str, default="6mer", help="DNABERT mode (e.g., 6mer, 3mer, nucleotide)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=16, help="Training and validation batch size")
    parser.add_argument("--log_fname", "--log-fname", dest="log_fname", type=str, default="training.log", help="Path to the log file")

    args = parser.parse_args()
    configure_logging(args.log_fname)
    train_lora_multilabel(args)
