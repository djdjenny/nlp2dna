import argparse
import logging
import os
import random
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)

MAX_POSITION_EMBEDDINGS = 514
SEED = 42


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

    def encode(self, sequence, max_length=MAX_POSITION_EMBEDDINGS):
        tokens = self.tokenize(sequence)
        ids = self.convert_tokens_to_ids(tokens)

        ids = [self.cls_token_id] + ids + [self.sep_token_id]

        if len(ids) > max_length:
            raise ValueError(f"Sequence is too long ({len(ids)} > {max_length})")

        attention_mask = [1] * len(ids)
        padding = max_length - len(ids)
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


class GenomeDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["sequence"]
        encoded = self.tokenizer.encode(sequence)

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }


class GenomeDatasetBPE(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["sequence"]

        return self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_POSITION_EMBEDDINGS,
        )


class DNAOverlappingCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.k = tokenizer.k
        self.mlm_probability = mlm_probability
        self.first_valid_kmer_id = len(tokenizer.special_tokens)

    def __call__(self, examples):
        input_ids = torch.stack([x["input_ids"] for x in examples])
        attention_mask = torch.stack([x["attention_mask"] for x in examples])

        labels = input_ids.clone()

        batch_size, _ = input_ids.shape
        device = input_ids.device
        valid_lengths = attention_mask.sum(dim=1)

        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            num_tokens = valid_lengths[b].item() - 2
            num_to_mask = int(num_tokens * self.mlm_probability)
            span_length = 2 * self.k - 1
            num_spans = max(1, round(num_to_mask / span_length)) if num_to_mask > 0 else 0
            max_start = num_tokens - span_length + 1

            if max_start < 1 or num_spans == 0:
                continue

            available_starts = list(range(1, max_start + 1))
            random.shuffle(available_starts)
            selected_spans = []

            for start in available_starts:
                span_end = start + span_length
                if all(
                    span_end <= selected_start or start >= selected_end
                    for selected_start, selected_end in selected_spans
                ):
                    selected_spans.append((start, span_end))
                    masked_indices[b, start:span_end] = True
                    if len(selected_spans) == num_spans:
                        break

        masked_indices &= attention_mask.bool()
        labels[~masked_indices] = -100

        r = torch.rand_like(input_ids, dtype=torch.float)
        mask_token_mask = masked_indices & (r < 0.8)
        input_ids[mask_token_mask] = self.tokenizer.mask_token_id

        random_token_mask = masked_indices & (r >= 0.8) & (r < 0.9)

        if random_token_mask.any():
            random_ids = torch.randint(
                self.first_valid_kmer_id,
                self.tokenizer.vocab_size,
                size=(random_token_mask.sum(),),
                device=device,
            )
            input_ids[random_token_mask] = random_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DNANonOverlappingCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.first_valid_kmer_id = len(tokenizer.special_tokens)

    def __call__(self, examples):
        input_ids = torch.stack([x["input_ids"] for x in examples])
        attention_mask = torch.stack([x["attention_mask"] for x in examples])

        labels = input_ids.clone()
        batch_size, _ = input_ids.shape
        device = input_ids.device
        valid_lengths = attention_mask.sum(dim=1)
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            num_tokens = valid_lengths[b].item() - 2
            num_to_mask = int(num_tokens * self.mlm_probability)
            if num_to_mask == 0:
                continue

            token_positions = list(range(1, num_tokens + 1))
            random.shuffle(token_positions)
            masked_indices[b, token_positions[:num_to_mask]] = True

        masked_indices &= attention_mask.bool()
        labels[~masked_indices] = -100

        r = torch.rand_like(input_ids, dtype=torch.float)
        mask_token_mask = masked_indices & (r < 0.8)
        input_ids[mask_token_mask] = self.tokenizer.mask_token_id

        random_token_mask = masked_indices & (r >= 0.8) & (r < 0.9)
        if random_token_mask.any():
            random_ids = torch.randint(
                self.first_valid_kmer_id,
                self.tokenizer.vocab_size,
                size=(random_token_mask.sum(),),
                device=device,
            )
            input_ids[random_token_mask] = random_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, checkpoint_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model, logger):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(f"Saved best model checkpoint: {self.checkpoint_path}")
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_config(vocab_size):
    return BertConfig(
        vocab_size=vocab_size,
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
    )


def build_components(mode, train_csv, val_csv, mlm_probability):
    if mode == "bpe":
        tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
        train_dataset = GenomeDatasetBPE(train_csv, tokenizer)
        val_dataset = GenomeDatasetBPE(val_csv, tokenizer)
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )
    else:
        k_map = {
            "nucleotide": 1,
            "3mer": 3,
            "6mer": 6,
            "3mer_no": 3,
            "6mer_no": 6,
        }
        if mode.endswith("_no"):
            tokenizer = DNANonOverlappingTokenizer(k=k_map[mode])
            collator_class = DNANonOverlappingCollator
        else:
            tokenizer = DNATokenizer(k=k_map[mode])
            collator_class = DNAOverlappingCollator
        train_dataset = GenomeDataset(train_csv, tokenizer)
        val_dataset = GenomeDataset(val_csv, tokenizer)
        collator = collator_class(
            tokenizer,
            mlm_probability=mlm_probability,
        )

    config = build_model_config(vocab_size=tokenizer.vocab_size)
    model = BertForMaskedLM(config)

    return train_dataset, val_dataset, collator, model


def train_dnabert(
    model,
    train_dataset,
    val_dataset,
    collator,
    output_dir,
    logger,
    epochs=3,
    batch_size=64,
    lr=1e-4,
    max_grad_norm=1.0,
    patience=5,
    min_delta=1e-4,
    device="cuda",
    model_title="dnabert",
):
    model = model.to(device)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * 0.10)
    warmup_start_lr = 1e-6
    min_lr = 1e-4

    def lr_lambda(step):
        if step < warmup_steps:
            progress = step / max(1, warmup_steps)
            current_lr = warmup_start_lr + (lr - warmup_start_lr) * progress
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
            current_lr = min_lr + (lr - min_lr) * cosine_factor
        return current_lr / lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_checkpoint_path = os.path.join(output_dir, f"{model_title}_best.pt")
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        checkpoint_path=best_checkpoint_path,
    )

    global_step = 0
    seen_sequences = 0
    losses = []
    model.train()

    stop_training = False
    for epoch in range(epochs):
        if stop_training:
            break

        epoch_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            epoch_loss += loss.item()

            seen_sequences += batch_size
            global_step += 1

            if global_step % 100 == 0:
                logger.info(
                    f"[epoch {epoch} | step {global_step}] " f"train_loss = {loss.item():.4f} seen = {seen_sequences}"
                )

            if global_step % 100 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        val_outputs = model(
                            input_ids=val_batch["input_ids"],
                            attention_mask=val_batch["attention_mask"],
                            labels=val_batch["labels"],
                        )
                        val_loss += val_outputs.loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                logger.info(f"[epoch {epoch} | step {global_step}] " f"val_loss = {avg_val_loss:.4f}")

                early_stopper(avg_val_loss, model, logger)
                if early_stopper.early_stop:
                    logger.info(f"Training stopped early at step {global_step}")
                    stop_training = True
                    break

                model.train()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} done. avg train loss = {avg_loss:.4f}")

        epoch_checkpoint = os.path.join(output_dir, f"{model_title}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), epoch_checkpoint)
        logger.info(f"Saved epoch checkpoint: {epoch_checkpoint}")

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        logger.info(f"Loaded best checkpoint: {best_checkpoint_path}")

    return losses


def create_parser():
    parser = argparse.ArgumentParser(description="Train DNABERT from notebook cell 25+ pipeline")
    parser.add_argument("--train-data", required=True, help="Path to training CSV file")
    parser.add_argument("--val-data", required=True, help="Path to validation CSV file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["bpe", "3mer", "6mer", "3mer_no", "6mer_no", "nucleotide"],
        help="Tokenizer/data mode",
    )
    parser.add_argument("--model-title", required=True, help="Model title used for outputs")
    parser.add_argument("--max-epochs", required=True, type=int, help="Maximum number of epochs")
    parser.add_argument("--batch-size", required=True, type=int, help="Batch size")
    parser.add_argument("--output-dir", required=True, help="Directory for logs and checkpoints")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Early stopping minimum delta")
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="MLM probability")

    return parser


def setup_logger(log_file_path):
    logger = logging.getLogger("dnabert_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = create_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.model_title}.log")
    logger = setup_logger(log_path)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting as requested.")
        sys.stderr.write("CUDA is not available. Exiting as requested.\n")
        raise SystemExit(1)

    device = "cuda"
    set_seed(SEED)

    logger.info(f"Starting training with mode={args.mode}, model_title={args.model_title}")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed fixed to {SEED}")

    train_dataset, val_dataset, collator, model = build_components(
        mode=args.mode,
        train_csv=args.train_data,
        val_csv=args.val_data,
        mlm_probability=args.mlm_probability,
    )

    losses = train_dnabert(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        output_dir=args.output_dir,
        logger=logger,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        device=device,
        model_title=args.model_title,
    )

    loss_path = os.path.join(args.output_dir, f"{args.model_title}_loss.txt")
    with open(loss_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(v) for v in losses))

    logger.info(f"Saved loss values to: {loss_path}")
    logger.info("Training complete")


if __name__ == "__main__":
    main()
