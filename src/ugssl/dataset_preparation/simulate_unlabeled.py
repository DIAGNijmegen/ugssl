"""Simulated a labeled+unlabeled situation, given a fully labeled dataset.

This assumes the source dir looks like:

source_dir/
    |
    imagesTr/
    labelsTr/

and will write to dest_dir as:

dest_dir/
    |
    imagesTr/
    labelsTr/
    imagesUl/

this also assumes IDs <500 are CT scans, otherwise MR.

"""
from argparse import ArgumentParser
from pathlib import Path
import random
import shutil
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main(source_dir: Path, dest_dir: Path, percentage_labeled: int):
    if not dest_dir.is_dir():
        print("Destination directory does not exist, making it now.")
        dest_dir.mkdir(parents=True, exist_ok=True)

    labeled_dest_dir = dest_dir / "imagesTr"
    labeled_label_dest_dir = dest_dir / "labelsTr"
    unlabeled_dest_dir = dest_dir / "imagesUl"
    labeled_dest_dir.mkdir(exist_ok=True)
    labeled_label_dest_dir.mkdir(exist_ok=True)
    unlabeled_dest_dir.mkdir(exist_ok=True)

    # Get all files in source dir
    files = list((source_dir / "imagesTr").glob("*.nii.gz"))
    print(f"Found {len(files)} cases")

    # Get ct/mr, ct = 1, mr = 0
    is_ct = []
    for filepath in files:
        id = filepath.stem.split("_")[1]
        is_ct.append(int(id) < 500)

    to_select = int((percentage_labeled / 100) * len(files))
    print(f"Selecting {to_select} cases for labeled subset")

    # Copy image and labels
    _, selected_files = train_test_split(files, test_size=to_select, stratify=is_ct)
    for image_path in tqdm(files, desc="Copying files"):
        label_name = image_path.name.replace("_0000", "")
        label_path = source_dir / "labelsTr" / label_name
        if image_path in selected_files:
            shutil.copyfile(image_path, labeled_dest_dir / image_path.name)
            shutil.copyfile(label_path, labeled_label_dest_dir / label_path.name)
        else:
            shutil.copyfile(image_path, unlabeled_dest_dir / image_path.name)

    # Copy dataset.json and update number of training cases
    shutil.copyfile(source_dir / "dataset.json", dest_dir / "dataset.json")
    with open(dest_dir / "dataset.json") as json_file:
        dataset_json = json.load(json_file)
    dataset_json["numTraining"] = to_select
    with open(dest_dir / "dataset.json", "w") as json_file:
        json.dump(dataset_json, json_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("dest_dir", type=str)
    parser.add_argument("--percentage_labeled", type=int, default=10)
    args = parser.parse_args()
    main(Path(args.source_dir), Path(args.dest_dir), args.percentage_labeled)
