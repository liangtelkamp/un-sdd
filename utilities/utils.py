# Evaluation utilities
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
import pandas as pd
import re
from typing import Dict, Any
import pycountry
from langdetect import detect, DetectorFactory, LangDetectException


def load_json_data(file_path):
    import json

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(data, file_path):
    import json

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def table_markdown(
    table_data, pii_model=None, sensitivity_key=None, kii_key=None, rows=5
):
    columns_data = table_data["columns"]
    column_samples = {}
    pii_reflection_key = f"pii_reflection_{pii_model}"
    pii_key = f"pii_detection_{pii_model}"
    for column_name, column_info in columns_data.items():
        if not all(x == "" for x in column_info["records"]):
            column_key = column_name

            if pii_model and column_info.get(pii_reflection_key):
                if column_info[pii_reflection_key] != "NON_SENSITIVE":
                    column_key += f" - {column_info[pii_key]}"
            elif pii_model and column_info.get(pii_key) != "None":
                column_key += f" - {column_info[pii_key]}"
            if (
                sensitivity_key
                and column_info.get(sensitivity_key)
                and column_info[sensitivity_key] != "NON_SENSITIVE"
            ):
                column_key += f" - sensitive"
            if kii_key and column_info.get(kii_key):
                column_key += f" - KII entity: {column_info[kii_key]}"

            if len(column_info["records"]) > rows:
                column_samples[column_key] = column_info["records"][:rows]
            elif len(column_info["records"]) < rows:
                # Add empty strings to make it 5
                column_samples[column_key] = column_info["records"] + [""] * (
                    rows - len(column_info["records"])
                )

    df = pd.DataFrame(column_samples)
    # Delete empty rows
    df = df[df.apply(lambda row: row.astype(str).str.strip().any(), axis=1)]
    markdown_table = df.to_markdown()
    return markdown_table


def evaluate_pii_detection(
    data: Dict,
    model_name: str,
    *,
    show_fps: bool = False,
    show_fns: bool = False,
) -> Dict:
    """Evaluate PII detection results.

    Parameters
    ----------
    data: Dict
        Dataset with predictions and ground truth.
    model_name: str
        Name of the model used for predictions.
    show_fps: bool, optional
        When True, print false positive examples.
    show_fns: bool, optional
        When True, print false negative examples.
    Returns
    -------
    Dict with precision, recall, f1 and accuracy.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    fps = []
    fns = []

    for table_name, table in data.items():
        for column_name, col in table.get("columns", {}).items():
            gt = col.get("sensitivity_gt")
            pred = col.get(f"pii_reflection_{model_name}")
            if gt is None or pred is None:
                continue
            gt_bin = int(gt)
            pred_bin = 0 if pred == "NON_SENSITIVE" else 1
            y_true.append(gt_bin)
            y_pred.append(pred_bin)
            if pred_bin == 1 and gt_bin == 0:
                fps.append((table_name, column_name, gt, pred, table))
            elif pred_bin == 0 and gt_bin == 1:
                fns.append((table_name, column_name, gt, pred, table))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    if show_fps and fps:
        print("\n### False Positives")
        fp_data = [(tname, cname, gt, pred) for tname, cname, gt, pred, table in fps]
        print(
            tabulate(
                fp_data,
                headers=["Table", "Column", "Ground Truth", "Prediction"],
                tablefmt="grid",
            )
        )
        # for tname, cname, gt, pred, table in fps:
        #     # print(f"\n**Table: {tname} | Column: {cname}**")
        #     print(table_markdown(table))

    if show_fns and fns:
        print("\n### False Negatives")
        fn_data = [(tname, cname, gt, pred) for tname, cname, gt, pred, table in fns]
        print(
            tabulate(
                fn_data,
                headers=["Table", "Column", "Ground Truth", "Prediction"],
                tablefmt="grid",
            )
        )
        # for tname, cname, gt, pred, table in fns:
        #     # print(f"\n**Table: {tname} | Column: {cname}**")
        #     print(table_markdown(table))

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def evaluate_non_pii_table(
    data: Dict,
    model_name: str,
    *,
    show_fps: bool = False,
    show_fns: bool = False,
) -> Dict:
    """Evaluate non-PII table classification with and without ISP guidance."""
    y_true: List[int] = []
    y_pred_isp: List[int] = []
    y_pred_no_isp: List[int] = []

    fps_isp = []
    fns_isp = []
    fps_no_isp = []
    fns_no_isp = []

    for table_name, table in data.items():
        meta = table.get("metadata", {})
        gt = meta.get("non_pii")
        pred_isp = meta.get(f"non_pii_{model_name}")
        pred_isp_expl = meta.get(f"non_pii_{model_name}_explanation")
        pred_no_isp = meta.get(f"non_pii_no_isp_{model_name}")
        pred_no_isp_expl = meta.get(f"non_pii_no_isp_{model_name}_explanation")
        if gt is None or pred_isp is None or pred_no_isp is None:
            continue

        gt_bin = 0 if gt == "NON_SENSITIVE" else 1
        pred_isp_bin = 0 if pred_isp == "NON_SENSITIVE" else 1
        pred_no_isp_bin = 0 if pred_no_isp == "NON_SENSITIVE" else 1

        y_true.append(gt_bin)
        y_pred_isp.append(pred_isp_bin)
        y_pred_no_isp.append(pred_no_isp_bin)

        if pred_isp_bin == 1 and gt_bin == 0:
            fps_isp.append((table_name, gt, pred_isp, pred_isp_expl, table))
        elif pred_isp_bin == 0 and gt_bin == 1:
            fns_isp.append((table_name, gt, pred_isp, pred_isp_expl, table))

        if pred_no_isp_bin == 1 and gt_bin == 0:
            fps_no_isp.append((table_name, gt, pred_no_isp, pred_no_isp_expl, table))
        elif pred_no_isp_bin == 0 and gt_bin == 1:
            fns_no_isp.append((table_name, gt, pred_no_isp, pred_no_isp_expl, table))

    precision_isp, recall_isp, f1_isp, _ = precision_recall_fscore_support(
        y_true, y_pred_isp, average="binary", zero_division=0
    )
    acc_isp = accuracy_score(y_true, y_pred_isp)
    precision_no, recall_no, f1_no, _ = precision_recall_fscore_support(
        y_true, y_pred_no_isp, average="binary", zero_division=0
    )
    acc_no = accuracy_score(y_true, y_pred_no_isp)

    if show_fps and (fps_isp or fps_no_isp):
        if fps_isp:
            print("\n### False Positives (with ISP)")
            fp_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fps_isp
            ]
            print(
                tabulate(
                    fp_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fps_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

        if fps_no_isp:
            print("\n### False Positives (without ISP)")
            fp_no_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fps_no_isp
            ]
            print(
                tabulate(
                    fp_no_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fps_no_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

    if show_fns and (fns_isp or fns_no_isp):
        if fns_isp:
            print("\n### False Negatives (with ISP)")
            fn_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fns_isp
            ]
            print(
                tabulate(
                    fn_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fns_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

        if fns_no_isp:
            print("\n### False Negatives (without ISP)")
            fn_no_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fns_no_isp
            ]
            print(
                tabulate(
                    fn_no_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fns_no_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

    return {
        "with_isp": {
            "precision": precision_isp,
            "recall": recall_isp,
            "f1": f1_isp,
            "accuracy": acc_isp,
        },
        "without_isp": {
            "precision": precision_no,
            "recall": recall_no,
            "f1": f1_no,
            "accuracy": acc_no,
        },
    }


def fetch_country(input_string: str) -> str:
    """Extract country name from a string with gpt-4o-mini"""
    from llm_model.model import Model

    prompt = f"""
    Extract country name from the following string, only return the country name:
    {input_string}
    """
    llm = Model("gpt-4o-mini")
    prediction = llm.generate(prompt)
    return prediction


# To make results deterministic (optional)
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """
    Detects the language code of the given text using langdetect.
    Returns the ISO 639-1 language code, e.g., 'en' for English.
    If detection fails, returns 'unknown'.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def import_csv_xlsx(file_path: str) -> pd.DataFrame:
    """
    Imports a CSV or XLSX file into a pandas DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, on_bad_lines="skip")
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# Example usage
if __name__ == "__main__":
    print(standardize_country("Somalia_survey_results_2023.xlsx"))
    print(standardize_country("us"))
    print(standardize_country("The Netherlands"))
    print(standardize_country("SOM"))
    print(standardize_country("unrelated_filename"))
