from itertools import product
import os
from typing import List
import numpy as np
import torch
from transformers import BertConfig, BertModel
import sys
import gc

MAX_POSITION_EMBEDDINGS = 514


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
        
        if "nucl" in model_name or "nucleotide" in model_name:
            return "nucleotide"
        elif "3mer_no" in model_name or "3mer-no" in model_name:
            return "3mer_no"
        elif "6mer_no" in model_name or "6mer-no" in model_name:
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
                chunk_embeddings[i : i + len(batch_sequences)] = cls_embeddings.cpu().numpy()
                del outputs, input_ids, attention_mask, encoded, cls_embeddings


        embeddings = []
        offset = 0
        for sequence_chunks in chunked_sequences:
            count = len(sequence_chunks)
            embeddings.append(chunk_embeddings[offset : offset + count].mean(axis=0))
            offset += count

        return np.vstack(embeddings)

# в итоге не использовала пока что 
class DNABertBPEExtractor:
    """Extractor for DNABERT BPE mode using GROVER tokenizer"""

    def __init__(self, name_model, device="cuda"):
        from transformers import AutoTokenizer
        self.device = device
        self.name_model = name_model
        self.mode = "bpe"

        # Check for model path in environment variable
        self.model_path = os.getenv("DNABERT_MODEL_PATH")
        if not self.model_path:
            raise ValueError(
                "DNABERT_MODEL_PATH environment variable not set. "
                "Please provide model weights path via environment variable."
            )

        self.tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")

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
            output_hidden_states=True,
        )

    def extract_embeddings(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i : i + batch_size]

                # Tokenize with HF tokenizer
                encoded = self.tokenizer(
                    batch_sequences,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=MAX_POSITION_EMBEDDINGS,
                    padding=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(embeddings)
