# from .sensitivityClassifier import SensitivityClassifier
from .utils import table_markdown

def detect_and_reflect_pii(table_data, generator):

    for column_name, col_data in table_data["columns"].items():
        sample_values = col_data['records']
        if not col_data.get(f"pii_detection_{generator.model_name}"):
            pii_entity = generator.classify_pii(column_name, sample_values)
            col_data[f"pii_detection_{generator.model_name}"] = pii_entity


    table_md = table_markdown(table_data, pii_model=generator.model_name)

    for column_name, col_data in table_data["columns"].items():
        if col_data.get(f"pii_reflection_{generator.model_name}") and col_data.get(f"pii_reflection_{generator.model_name}") in ['NON_SENSITIVE', 'MEDIUM_SENSITIVE', 'HIGH_SENSITIVE']:
            continue
        pii_entity = col_data[f"pii_detection_{generator.model_name}"] 
        sensitivity = generator.classify_sensitive_pii(column_name, table_md, pii_entity)
        col_data[f"pii_reflection_{generator.model_name}"] = sensitivity

    return table_data
