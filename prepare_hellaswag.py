import os
import requests

def download_hellaswag():
    """Download HellaSwag validation dataset and save locally"""

    # URL for the validation dataset
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

    # Create data directory if it doesn't exist
    data_dir = "hellaswag_data"
    os.makedirs(data_dir, exist_ok=True)

    # Output file path
    output_path = os.path.join(data_dir, "hellaswag_val.jsonl")

    print(f"Downloading HellaSwag validation dataset from:")
    print(f"  {url}")
    print(f"Saving to: {output_path}")

    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise error if download failed

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

    # Count number of examples
    num_examples = len(response.text.strip().split('\n'))

    print(f"✓ Downloaded successfully!")
    print(f"  {num_examples} examples")
    print(f"  {len(response.text)} bytes")

if __name__ == "__main__":
    download_hellaswag()
