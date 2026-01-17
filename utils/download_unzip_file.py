from pathlib import Path
import requests
import zipfile


def download_file(url: str, save_path: Path) -> None:
    file = requests.get(url)
    open(save_path, "wb").write(file.content)


def unzip_file(zip_file):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all files from ", zip_file)
    except:
        print("Invalid file: ", zip_file)


def download_unzip_file(url: str, save_path: Path):
    if not save_path.exists():
        download_file(url, save_path)

    unzip_file(save_path)
