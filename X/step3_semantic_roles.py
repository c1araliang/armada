"""
Step 3 helper: semantic role labeling for group-role extraction.

Uses a Hugging Face token-classification SRL model that expects a special [V]
marker immediately before the predicate token.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


SRL_MODEL_NAME = "dannashao/bert-base-uncased-finetuned-advanced-srl_arg"
PATIENT_LABELS = {"ARG1", "ARG1-DSP", "ARG2", "ARG3", "ARG4", "ARG5"}


class SrlRoleLabeler:
    def __init__(self, model_name: str = SRL_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _predicate_candidates(self, doc, predicate_indices: set[int] | None = None):
        for token in doc:
            if predicate_indices is not None and token.i not in predicate_indices:
                continue
            if token.pos_ == "VERB":
                yield token
                continue
            if token.tag_ in {"VBN", "VBG"} and token.pos_ in {"ADJ", "AUX"}:
                yield token
                continue
            if token.pos_ == "ADJ" and any(child.dep_ == "auxpass" for child in token.children):
                yield token

    def _words_with_marker(self, doc, predicate_i: int):
        words = []
        orig_index_by_word = []
        for i, token in enumerate(doc):
            if i == predicate_i:
                words.append("[V]")
                orig_index_by_word.append(None)
            words.append(token.text)
            orig_index_by_word.append(i)
        return words, orig_index_by_word

    def _predict_word_labels_batch(self, doc, predicate_indices: list[int]) -> list[dict[int, str]]:
        words_batch = []
        index_maps = []
        for predicate_i in predicate_indices:
            words, orig_index_by_word = self._words_with_marker(doc, predicate_i)
            words_batch.append(words)
            index_maps.append(orig_index_by_word)

        batch_encoding = self.tokenizer(
            words_batch,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in batch_encoding.items()}

        with torch.no_grad():
            logits_batch = self.model(**encoded).logits

        id2label = self.model.config.id2label
        all_labels = []
        for batch_idx, orig_index_by_word in enumerate(index_maps):
            word_ids = batch_encoding.encodings[batch_idx].word_ids
            logits = logits_batch[batch_idx]
            word_logits: dict[int, list[torch.Tensor]] = defaultdict(list)
            for token_pos, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                orig_index = orig_index_by_word[word_id]
                if orig_index is None:
                    continue
                word_logits[orig_index].append(logits[token_pos].detach().cpu())

            labels = {}
            for orig_index, pieces in word_logits.items():
                avg_logits = torch.stack(pieces).mean(dim=0)
                raw_label = id2label[int(avg_logits.argmax().item())]
                label = raw_label.replace("C-", "").replace("R-", "")
                labels[orig_index] = label
            all_labels.append(labels)
        return all_labels

    def annotate(self, doc, predicate_indices: set[int] | None = None) -> list[dict]:
        frames = []
        predicates = list(self._predicate_candidates(doc, predicate_indices=predicate_indices))
        if not predicates:
            return frames

        label_maps = self._predict_word_labels_batch(doc, [p.i for p in predicates])
        for predicate, labels in zip(predicates, label_maps):
            frames.append({
                "predicate_i": predicate.i,
                "predicate_text": predicate.text,
                "predicate_lemma": predicate.lemma_.lower(),
                "labels": labels,
            })
        return frames
