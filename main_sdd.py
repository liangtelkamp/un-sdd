"""main_sdd.py: Main script for the SDD Pipeline."""

import os
from datetime import datetime
from pathlib import Path
import json
import logging
import logging.config
import sys
import dotenv  # type: ignore
from utils.processing import DataSampler
from utils.main_config import ISP_DEFAULT, OUTPUT_DIR
from classifiers.pii_classifier import PIIClassifier  # type: ignore
from classifiers.non_pii_classifier import NonPIIClassifier  # type: ignore
from classifiers.pii_reflection_classifier import PIIReflectionClassifier  # type: ignore
import pandas as pd  # type: ignore
from models.sdd_report import SDDReport


def table_markdown(sdd_report: SDDReport):
    """Generate a markdown table from the report."""
    columns_data = sdd_report.columns
    column_samples = {}

    # Build dict of column -> list of values
    for column in columns_data:
        if column.pii.get("entity_type") != "None":
            column_key = f"{column.column_name} - {column.pii.get('entity_type')}"
        else:
            column_key = column.column_name
        column_samples[column_key] = column.sample_values

    # Find the maximum number of samples among all columns
    max_len = max(len(v) for v in column_samples.values())

    # Pad shorter columns with empty strings
    for key, values in column_samples.items():
        if len(values) < max_len:
            column_samples[key] = values + [""] * (max_len - len(values))

    # Create DataFrame and return as markdown
    table_data = pd.DataFrame(column_samples)
    return table_data.to_markdown(index=False)


if __name__ == "__main__":
    # ===== Logging =====
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    else:
        logging.basicConfig(
            filename="logs/sdd.log",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    logger = logging.getLogger(__name__)
    logger.propagate = True

    # ===== Environment Variables =====
    dotenv.load_dotenv()

    # ===== CKAN Client =====
    csv_file_name = "csv_file_name.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_file_name)

    # Check if sdd_report is already in iutput
    output_path = os.path.join(OUTPUT_DIR, f"{csv_file_name}_sdd_report.json")
    if os.path.exists(output_path):
        logger.info("SDD Report already exists in the output directory")
        report = SDDReport.from_json(output_path)
        sys.exit(1)
    else:
        logger.info("SDD Report does not exist in the output directory")

    # ===== Preprocessing & Sampling =====
    sampler = DataSampler()
    dfs_by_sheet = sampler.sample_from_local(
        file_path=csv_path,
    )  # returns dict: sheet_name -> df

    reports = []  # List to hold multiple reports if needed

    for sheet_name, df in dfs_by_sheet.items():
        if "readme" in sheet_name.lower() or "instrucciones" in sheet_name.lower():
            logger.info("Skipping readme sheet")
            continue
        logger.info("Processing sheet: %s", sheet_name)

        report = SDDReport(
            resource_id="csv_file_name",
            file_name=csv_file_name,
            file_url="",
            sheet_name=sheet_name,
            processing_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            processing_success=True,
            n_records=len(df),
            n_columns=len(df.columns),
        )

        # ===== PII Detection =====
        if report.pii_classifier_model is None:
            logger.info("Starting PII Detection for sheet '%s'...", sheet_name)
            pii_detector = PIIClassifier(model_name="gpt-4.1-nano")
            report = pii_detector.classify_df(df=df, report=report)
        else:
            logger.info(
                "PII Detection already performed, skipping sheet '%s'", sheet_name
            )

        # ===== PII Reflection Detection =====
        if report.pii_reflection_model is None:
            logger.info(
                "Starting PII Reflection Detection for sheet '%s'...", sheet_name
            )
            pii_reflection_classifier = PIIReflectionClassifier(
                model_name="gpt-4.1-nano"
            )
            report = pii_reflection_classifier.classify_df(
                table_markdown=table_markdown(report), report=report
            )
        else:
            logger.info(
                "PII Reflection Detection already performed, skipping sheet '%s'",
                sheet_name,
            )

        # ===== Non-PII Classification =====
        if report.non_pii is None:
            print(f"NON-PII CLASSIFICATION PERFORMED FOR {sheet_name}")
            logger.info("Starting Non-PII Classification for sheet '%s'...", sheet_name)
            non_pii_classifier = NonPIIClassifier(model_name="gpt-4.1-nano")
            report = non_pii_classifier.classify(
                table_markdown=table_markdown(report), report=report, isp=ISP_DEFAULT
            )
        else:
            print(f"NON-PII CLASSIFICATION ALREADY PERFORMED FOR {sheet_name}")
            logger.info(
                "Non-PII Classification already performed, skipping sheet '%s'",
                sheet_name,
            )
        reports.append(report.to_dict())

    # ===== Save report =====
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(reports, indent=2))
    logger.info("Report saved for file '%s' at %s", csv_path, output_path)
