# ============================================================
# Data loading and basic inspection helpers
# ============================================================

import glob
from pathlib import Path

import pandas as pd

from src.config import REQUIRED_FILES, SENSITIVE_ATTRS


def has_required_oulad_files(path):
    path = Path(path)
    return path.exists() and all((path / name).exists() for name in REQUIRED_FILES)


def resolve_data_dir(local_dir="data", colab_dir="/content/oulad", use_kagglehub=True):
    """
    Resolve the data directory: local first, then KaggleHub, then manual Colab path.
    Returns a Path object pointing to the data directory.
    """
    local_data_dir = Path(local_dir)
    colab_data_dir = Path(colab_dir)

    if has_required_oulad_files(local_data_dir):
        print("Using local data directory:", local_data_dir.resolve())
        return local_data_dir

    if use_kagglehub:
        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(
                "anlgrbz/student-demographics-online-education-dataoulad"
            )
            data_dir = Path(dataset_path)
            print("Downloaded dataset to:", data_dir)
            return data_dir
        except Exception as e:
            print("KaggleHub download failed. Falling back to manual /content/oulad directory.")
            print("Error:", e)

    colab_data_dir.mkdir(parents=True, exist_ok=True)
    print("Using manual data directory:", colab_data_dir)
    return colab_data_dir


def find_file(filename, root_dir):
    """Find a file recursively under root_dir."""
    matches = list(Path(root_dir).rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {root_dir}. "
            "Please check whether the OULAD CSV files are uploaded or downloaded correctly."
        )
    return matches[0]


def read_oulad_csv(filename, data_dir):
    path = find_file(filename, data_dir)
    print(f"Reading {filename} from {path}")
    return pd.read_csv(path)


def load_all_tables(data_dir):
    """
    Load all 7 OULAD CSV files.
    Returns a dict with keys: courses, assessments, student_assess,
    student_info, student_reg, student_vle, vle.
    """
    data_dir = Path(data_dir)
    tables = {
        "courses":        read_oulad_csv("courses.csv", data_dir),
        "assessments":    read_oulad_csv("assessments.csv", data_dir),
        "student_assess": read_oulad_csv("studentAssessment.csv", data_dir),
        "student_info":   read_oulad_csv("studentInfo.csv", data_dir),
        "student_reg":    read_oulad_csv("studentRegistration.csv", data_dir),
        "student_vle":    read_oulad_csv("studentVle.csv", data_dir),
        "vle":            read_oulad_csv("vle.csv", data_dir),
    }
    print("\nShapes:")
    for name, df in tables.items():
        print(f"  {name}: {df.shape}")
    return tables


def inspect_tables(student_info, courses, sensitive_attrs=None):
    """Print basic distributions for the main tables."""
    if sensitive_attrs is None:
        sensitive_attrs = SENSITIVE_ATTRS

    print("\nstudentInfo columns:")
    print(student_info.columns.tolist())

    print("\nFinal result distribution:")
    print(student_info["final_result"].value_counts(dropna=False))

    print("\nSensitive attribute distributions:")
    for attr in sensitive_attrs:
        print(f"\n{attr}")
        print(student_info[attr].value_counts(dropna=False))

    print("\nCourse presentation lengths:")
    print(courses.head())
