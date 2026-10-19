from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "Qwen/Qwen3-Reranker-0.6B"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen3-Reranker-0.6B"
MAX_LENGTH = 2048     # your chunks are ~512 tokens
BATCH_SIZE = 16       # lower to 8 if you run out of VRAM

TASK = "Given a financial question, retrieve SEC filing passages that answer it"

# Downloads into MODEL_DIR on the first run and skips files already there on later runs.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, attn_implementation="sdpa"
).cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)


def format_instruction(query, doc, instruction=TASK):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation="longest_first", return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    return {k: v.to(model.device) for k, v in inputs.items()}


@torch.no_grad()
def score_pairs(pairs):
    logits = model(**process_inputs(pairs)).logits[:, -1, :]
    two = torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1)
    return torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp().tolist()


def rerank(query, docs, top_k=5):
    scores = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        scores += score_pairs([format_instruction(query, d) for d in batch])
    ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    q = "What were the main risk factors Apple reported?"
    docs = [
        "Item 1A. Risk Factors: The Company's business depends on global supply "
        "chains and component availability.",
        "The company's headquarters are located in Cupertino, California.",
        "Competition in the smartphone market and foreign exchange rate changes "
        "may affect results.",
    ]
    for doc, s in rerank(q, docs, top_k=3):
        print(f"{s:.3f}  {doc[:80]}")