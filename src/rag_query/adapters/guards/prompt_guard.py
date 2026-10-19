"""Prompt Guard 2 query guard (prompt-injection / jailbreak detection).

Runs the local ``Llama-Prompt-Guard-2-86M`` classifier (an mDeBERTa-v2 model).
Prompt Guard 2 is a *binary* classifier (``BENIGN`` / ``MALICIOUS``); the label
set is read from the checkpoint, so the older ternary Prompt Guard 1
(``BENIGN`` / ``INJECTION`` / ``JAILBREAK``) also works.

We treat ``1 - P(BENIGN)`` as the malicious score and block when it crosses the
threshold. The model loads lazily on first use so importing the pipeline (and
unit tests) never pays for it.
"""

from __future__ import annotations

from pathlib import Path

from rag_query.domain.models import GuardVerdict
from rag_query.logging import get_logger

log = get_logger(__name__)


class PromptGuardModel:
    """Thin wrapper around the transformers classifier."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.to(device).eval()

        raw_labels = getattr(self._model.config, "id2label", {}) or {}
        labels = {int(k): str(v) for k, v in raw_labels.items()}
        if not labels or all(value.startswith("LABEL_") for value in labels.values()):
            # The released Prompt Guard 2 config omits id2label, so transformers
            # fills in LABEL_0/LABEL_1. The checkpoint is binary with class 0 =
            # benign and class 1 = malicious; without this mapping every benign
            # query would look 100% malicious.
            count = int(getattr(self._model.config, "num_labels", 0) or len(labels))
            if count == 2:
                labels = {0: "BENIGN", 1: "MALICIOUS"}
            elif count == 3:
                labels = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}
        self._labels = labels or {0: "BENIGN", 1: "MALICIOUS"}
        # Prompt Guard 2 is binary (BENIGN/MALICIOUS); Prompt Guard 1 is ternary
        # (BENIGN/INJECTION/JAILBREAK). Read the checkpoint's own label map and
        # never re-order it: re-labelling weights from the outside is guesswork
        # and silently turns a benign classifier into a false-positive machine.
        benign = next(
            (label for label in self._labels.values() if label.upper() == "BENIGN"),
            None,
        )
        if benign is None:
            benign = self._labels.get(0, "LABEL_0")
            log.warning(
                "prompt-guard checkpoint %s has no BENIGN label (%s); "
                "falling back to label 0 as benign",
                model_dir,
                self._labels,
            )
        self._benign_label = benign
        self._malicious_labels = tuple(
            label for label in self._labels.values() if label != self._benign_label
        )
        if len(self._labels) != 2:
            log.warning(
                "prompt-guard checkpoint %s has %d labels (%s); the supported "
                "binary Prompt Guard 2 checkpoint has 2. Verify results before "
                "enabling RAG_GUARD_ENABLED in production.",
                model_dir,
                len(self._labels),
                self._labels,
            )
        log.info("loaded prompt-guard %s labels=%s device=%s", model_dir, self._labels, device)

    @property
    def benign_label(self) -> str:
        return self._benign_label

    @property
    def malicious_labels(self) -> tuple[str, ...]:
        return self._malicious_labels

    def score(self, text: str) -> dict[str, float]:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**encoded).logits
        probs = self._torch.softmax(logits, dim=-1)[0].tolist()
        return {self._labels.get(i, f"label_{i}"): round(float(p), 6) for i, p in enumerate(probs)}


class PromptGuardQueryGuard:
    """Adapts :class:`PromptGuardModel` to the :class:`QueryGuard` port."""

    name = "prompt_guard"

    def __init__(self, model: PromptGuardModel, *, threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold

    def check(self, text: str) -> GuardVerdict:
        scores = self._model.score(text)
        benign = scores.get(self._model.benign_label, 0.0)
        malicious = round(max(0.0, 1.0 - benign), 6)
        allowed = malicious < self._threshold
        labels = tuple(
            label
            for label in self._model.malicious_labels
            if scores.get(label, 0.0) >= self._threshold
        )
        reason = "" if allowed else f"malicious probability {malicious:.3f} >= {self._threshold}"
        return GuardVerdict(
            allowed=allowed,
            guard=self.name,
            reason=reason,
            scores=scores,
            labels=labels,
        )
