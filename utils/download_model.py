import json
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()
DEFAULT_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("MODELS_DIR") or os.path.expanduser("~/.cache/huggingface")


def write_download_info(target_dir, model_id, token):
    """Record which exact revision was downloaded, and under which licence.

    Licences are irrevocable for the copy you already hold, but a publisher
    can relicense future releases or take a repo down. Without the commit
    hash there is no way to prove later which version arrived here, and on
    what terms. The file travels with the weights, which is the point.
    """
    info = {
        "repo_id": model_id,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "revision": None,
        "license": None,
        "last_modified": None,
    }

    try:
        meta = HfApi().model_info(repo_id=model_id, token=token)
        info["revision"] = meta.sha
        info["last_modified"] = str(getattr(meta, "last_modified", None))
        card = getattr(meta, "card_data", None)
        if card is not None:
            # card_data behaves like a dict but is not always one.
            info["license"] = getattr(card, "license", None) or (
                card.get("license") if hasattr(card, "get") else None
            )
    except Exception as e:  # noqa: BLE001 - bookkeeping must never break the download
        print(f"Notice: Could not determine revision ({e})")

    path = os.path.join(target_dir, "download_info.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)

    print(f"Revision:  {info['revision']}")
    print(f"License:   {info['license']}")
    print(f"Log:       {path}")


def download_hf_model(model_id, cache_dir=DEFAULT_CACHE_DIR):
    """Download a model repository from Hugging Face into target directory."""
    print("\n--- Hugging Face Model Downloader ---")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token or token == "YOUR_TOKEN_HERE":
        print(
            "Notice: No HF_TOKEN found in .env. Public models will download anyway."
        )
    else:
        print("Authentication: HF_TOKEN loaded.")

    print(f"Model:  {model_id}")
    print(f"Target: {cache_dir}")

    try:
        # Use local_dir to download direct files without symlinks (prevents WinError 1314 on Windows)
        target_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
        os.makedirs(target_dir, exist_ok=True)

        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            token=token,
            max_workers=8,
        )

        print("\nSuccess! The model was saved to:")
        print(local_path)

        write_download_info(target_dir, model_id, token)

        return local_path

    except Exception as e:  # noqa: BLE001 - report download failure instead of raising
        print(f"\nDownload error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_model = sys.argv[1]
    else:
        target_model = input(
            "Which model do you want to download? (e.g. microsoft/VibeVoice-ASR-HF): "
        )

    if target_model.strip():
        download_hf_model(target_model.strip())
    else:
        print("No model name provided.")
