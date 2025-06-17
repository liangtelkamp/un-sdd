
def load_json_data(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_json_data(data, file_path):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def table_markdown(table_data, pii_model=None, sensitivity_key=None, kii_key=None, rows=5):
    import pandas as pd

    columns_data = table_data['columns']
    column_samples = {}
    pii_reflection_key = f"pii_reflection_{pii_model}"
    pii_key = f"pii_detection_{pii_model}"
    for column_name, column_info in columns_data.items():
        if not all(x == '' for x in column_info['records']):
            column_key = column_name
            
            if pii_model and column_info.get(pii_reflection_key):
                if column_info[pii_reflection_key] != 'NON_SENSITIVE':
                    column_key += f" - {column_info[pii_key]}"
            elif pii_model and column_info.get(pii_key) != 'None':
                column_key += f" - {column_info[pii_key]}"
            if sensitivity_key and column_info.get(sensitivity_key) and column_info[sensitivity_key] != 'NON_SENSITIVE':
                column_key += f" - sensitive"
            if kii_key and column_info.get(kii_key):
                column_key += f" - KII entity: {column_info[kii_key]}"

            if len(column_info['records']) > rows:
                column_samples[column_key] = column_info['records'][:rows]
            elif len(column_info['records']) < rows:
                # Add empty strings to make it 5
                column_samples[column_key] = column_info['records'] + [""] * (rows - len(column_info['records']))
    
    df = pd.DataFrame(column_samples)
    # Delete empty rows
    df = df[df.apply(lambda row: row.astype(str).str.strip().any(), axis=1)]
    markdown_table = df.to_markdown()
    return markdown_table


