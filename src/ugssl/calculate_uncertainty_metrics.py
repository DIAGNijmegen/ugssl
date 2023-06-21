"""Script which calculates uncertainty metrics for a given directory
with predictions

This assumes that `prediction_path` has the following structure:
    fold_0
        case_1.nii.gz
        case_1.npz
        case_2.nii.gz
        case_2.npz
        ...
    fold_1
    ...

where nii.gz are segmentation masks and npz are raw softmax outputs
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import SimpleITK as sitk

from ugssl.metrics import PairwiseDSC


def load_nifti(path):
    image_sitk = sitk.ReadImage(str(path))
    image = sitk.GetArrayFromImage(image_sitk)
    return image


def is_complete(predictions_dir, case_id):
    for fold_idx in range(5):
        if not (predictions_dir / f"fold_{fold_idx}" / f"{case_id}.nii.gz").is_file():
            return False

    return True


def load_predictions(predictions_dir, case_id):
    predictions = []
    for fold_idx in range(5):
        path = predictions_dir / f"fold_{fold_idx}" / f"{case_id}.nii.gz"
        predictions.append(load_nifti(path))

    return predictions


def main(predictions_dir, csv_path, pkl_dir, n_labels, verbose=False):
    if pkl_dir and not pkl_dir.is_dir():
        pkl_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.is_file():
        df = pd.read_csv(csv_path, index_col=0)
        csv_dict = df.to_dict("index")
    else:
        csv_dict = {}

    metrics = {"pairwise_dsc": PairwiseDSC([i for i in range(n_labels + 1)])}

    for case_path in (predictions_dir / "fold_0").glob("*.nii.gz"):
        case_id = case_path.name[:-7]

        if case_id in csv_dict:
            if verbose:
                print(f"Skipping previously completed case {case_id}")
            continue

        if not is_complete(predictions_dir, case_id):
            if verbose:
                print(f"Skipping incomplete case {case_id}")
            continue

        predictions = load_predictions(predictions_dir, case_id)
        csv_dict[case_id] = {}
        for metric_name, metric in metrics.items():
            if pkl_dir:
                pkl_path = pkl_dir / f"{case_id}_{metric_name}.pkl"
            result = metric(predictions, pkl_path=pkl_path)

            for value_label, value in result.items():
                csv_dict[case_id][value_label] = value

        csv_df = pd.DataFrame.from_dict(csv_dict, orient="index")
        csv_df.to_csv(csv_path, index_label="case_id")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("predictions_dir", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--pkl_dir", type=str, required=False)
    parser.add_argument("--n_labels", type=int, default=15, required=False)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.pkl_dir:
        args.pkl_dir = Path(args.pkl_dir)

    main(
        Path(args.predictions_dir),
        Path(args.csv_path),
        args.pkl_dir,
        args.n_labels,
        verbose=args.verbose,
    )