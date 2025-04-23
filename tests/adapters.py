#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import gzip
import random
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        self.seq_length = seq_length
        docs = []
        open_fn = gzip.open if str(dataset_path).endswith(".gz") else open
        with open_fn(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                response = obj["response"]
                text = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{prompt}\n\n"
                    f"### Response:\n{response}"
                )
                docs.append(text)
        if shuffle:
            random.shuffle(docs)

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        full_ids = []
        for doc in docs:
            if bos_id is not None:
                full_ids.append(bos_id)
            ids = tokenizer.encode(doc, add_special_tokens=False)
            full_ids.extend(ids)
            if eos_id is not None:
                full_ids.append(eos_id)

        total_seqs = len(full_ids) // seq_length
        self.full_ids = full_ids
        self.token_ids = full_ids[: total_seqs * seq_length]
        self.num_sequences = total_seqs

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError

        start = idx * self.seq_length
        end = start + self.seq_length
        input_ids = self.token_ids[start:end]
        labels    = self.full_ids[start+1 : end+1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }
    
def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    pattern = r"[Tt]he\s+correct\s+answer\s+is\s*([ABCD])\b"
    match = re.search(pattern, model_output)
    if match:
        return match.group(1).upper()

    stripped = model_output.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped

    match = re.search(r'\b([ABCD])\b', model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', model_output)
    
    if numbers:
        return numbers[-1]
    else:
        return None
    raise NotImplementedError


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
