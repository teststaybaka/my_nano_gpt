"""Download all eval datasets used by eval_short.py and eval_long.py:

- HellaSwag validation  → hellaswag_data/hellaswag_val.jsonl
- LAMBADA test set      → lambada_data/lambada_test.jsonl
- PG19 validation books → pg19_data/pg19_val.jsonl

Run once per machine. Outputs are tokenizer-agnostic JSONL so the eval
scripts can tokenize at run time with whatever encoder the model uses.
"""
import os
import json
import argparse
import requests

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
LAMBADA_URL = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"


def download_jsonl(url, output_dir, filename):
    """Download a JSONL file directly from a URL."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    print(f"  GET {url}")
    print(f"  →   {output_path}")
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    num_examples = len(response.text.strip().split('\n'))
    print(f"  ✓ {num_examples} examples, {len(response.text)/1e6:.2f}MB\n")


def download_hellaswag():
    print("HellaSwag (commonsense completion, 4-way MCQ, short)")
    download_jsonl(HELLASWAG_URL, "hellaswag_data", "hellaswag_val.jsonl")


def download_lambada():
    print("LAMBADA (last-word cloze, short)")
    download_jsonl(LAMBADA_URL, "lambada_data", "lambada_test.jsonl")


def download_pg19():
    print("PG19 (Project Gutenberg long-form books, perplexity, long)")
    from datasets import load_dataset
    output_dir = "pg19_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pg19_val.jsonl")
    print(f"  loading emozilla/pg19 validation split via HuggingFace datasets…")
    ds = load_dataset("emozilla/pg19", split="validation")
    print(f"  →   {output_path}")
    total_chars = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(ds):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            total_chars += len(example['text'])
            if (i + 1) % 10 == 0:
                print(f"    wrote {i+1}/{len(ds)} books, {total_chars/1e6:.1f}M chars so far")
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  ✓ {len(ds)} books, {total_chars/1e6:.1f}M chars, {size_mb:.1f}MB file\n")


DATASETS = {
    "hellaswag": download_hellaswag,
    "lambada":   download_lambada,
    "pg19":      download_pg19,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets", nargs="*", default=list(DATASETS.keys()),
        help=f"Subset to download. Default = all. Choices: {list(DATASETS.keys())}",
    )
    args = parser.parse_args()
    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset: {name!r}; choices: {list(DATASETS.keys())}")
        DATASETS[name]()
    print("Done.")


if __name__ == "__main__":
    main()
