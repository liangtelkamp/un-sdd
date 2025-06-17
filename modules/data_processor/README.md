# Data Processor

## Example

```python
from modules.data_processor import DataLoader

# Initialize loader
loader = DataLoader(max_records_per_column=20)

# Load single file
data = loader.load_single_file("path/to/US_data.csv")

# Load entire folder
data = loader.load_folder("path/to/data_folder")

# Load any input (file or folder)
data = loader.load_data("path/to/input")

# Save processed data
loader.save_data(data, "output.json")
```

