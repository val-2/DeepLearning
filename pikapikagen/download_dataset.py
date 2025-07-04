import os
import urllib.request
import zipfile

def reporthook(block_num, block_size, total_size):
    if block_num % 16384 == 0:
        print(f"Downloading... {block_num * block_size / (1024 * 1024):.2f} MB")


def download_dataset_if_not_exists():
    dataset_dir = "dataset"
    pokedex_main_dir = os.path.join(dataset_dir, "pokedex-main")
    zip_url = "https://github.com/cristobalmitchell/pokedex/archive/refs/heads/main.zip"
    zip_path = "pokedex_main.zip"

    # Check if dataset/pokedex-main already exists
    if os.path.exists(pokedex_main_dir):
        print(f"{pokedex_main_dir} already exists. Skipping download.")
        return

    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    # Download the zip file
    print("Downloading dataset...")
    urllib.request.urlretrieve(zip_url, zip_path, reporthook)
    print("Download complete.")

    # Extract the zip file into the dataset directory
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction complete.")

    # Optionally, remove the zip file after extraction
    os.remove(zip_path)

if __name__ == "__main__":
    download_dataset_if_not_exists()
