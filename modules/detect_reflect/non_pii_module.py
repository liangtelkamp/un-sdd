# from .generation_utils import SensitivityClassifier
from .utils import table_markdown
import json
import re
import os
from pathlib import Path

# Laad ISP-regels bij moduleimport
module_dir = Path(__file__).parent
with open(module_dir / "isps.json", "r") as f:
    ISP_DATA = json.load(f)

def get_isp_by_country(table_name):
    table_name = table_name.lower()
    country_keywords = []

    for isp_name, isp_content in ISP_DATA.items():
        location = isp_content.get("location", "").lower()
        if location and re.search(rf"\\b{re.escape(location)}\\b", table_name):
            return isp_name, isp_content

        # Fallback: probeer ook match op de sleutel
        if location in table_name or location.replace(" ", "") in table_name:
            return isp_name, isp_content

    return 'default', ISP_DATA["default"]


def detect_non_pii(table_data, generator, fname, method='column'):
    isp_name, isp_used = None, None  # Ensure variables are always defined

    # Initialize metadata if not present
    if not table_data.get("metadata"):
        table_data["metadata"] = {}
    
    # Get ISP data
    isp_name, isp_used = get_isp_by_country(fname)
    table_data["metadata"]["isp_used"] = isp_name

    # table_md = table_markdown(table_data, pii_model=generator.model_name)
    table_md = table_markdown(table_data, pii_model=None)

    if method == 'column':
        if "columns" not in table_data:
            table_data["columns"] = {}
            
        for column_name, col_data in table_data["columns"].items():
            # Initialize column metadata if not present
            if not isinstance(col_data, dict):
                col_data = {}
                table_data["columns"][column_name] = col_data
                
            reflection_key = f'non_pii_{generator.model_name}'
            explanation_key = f'non_pii_{generator.model_name}_explanation'
            
            # Only process if not already done or if there was an error
            if not col_data.get(reflection_key) or col_data.get(reflection_key) == 'ERROR_GENERATION':
                sensitivity, explanation = generator.classify_sensitive_non_pii(column_name, table_md, isp_used)
                col_data[reflection_key] = sensitivity
                col_data[explanation_key] = explanation
    
    elif method == 'table':
        # Initialize required metadata keys
        non_pii_key = f'non_pii_{generator.model_name}'
        non_pii_explanation_key = f'non_pii_{generator.model_name}_explanation'
        non_pii_no_isp_key = f'non_pii_no_isp_{generator.model_name}'
        non_pii_no_isp_explanation_key = f'non_pii_no_isp_{generator.model_name}_explanation'
        
        # Process with ISP if not already done
        if non_pii_key not in table_data['metadata']:
            sensitivity, explanation = generator.classify_sensitive_non_pii_table(table_md, isp=isp_used)
            table_data['metadata'][non_pii_key] = sensitivity
            table_data['metadata'][non_pii_explanation_key] = explanation
            
            # Process without ISP
            sensitivity, explanation = generator.classify_sensitive_non_pii_table(table_md, isp=None)
            table_data['metadata'][non_pii_no_isp_key] = sensitivity
            table_data['metadata'][non_pii_no_isp_explanation_key] = explanation

    return table_data
