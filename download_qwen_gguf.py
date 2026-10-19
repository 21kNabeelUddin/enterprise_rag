"""Download the Qwen3.5-9B Q4_K_M GGUF used by llama.cpp."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "bartowski/Qwen_Qwen3.5-9B-GGUF"
FILENAME = "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
    )
    print("saved:", path)


if __name__ == "__main__":
    main()
