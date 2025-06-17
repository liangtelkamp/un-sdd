"""
Example usage of the data processor module.
"""

import json
from pathlib import Path
from data_loader import DataLoader
from country_utils import CountryExtractor


def main():
    """Demonstrate the data processor functionality."""
    
    # Initialize components
    loader = DataLoader(max_records_per_column=10)
    country_extractor = CountryExtractor()
    
    print("=== Data Processor Example ===\n")
    
    # Example 1: Country extraction from filenames
    print("1. Country extraction from filenames:")
    test_filenames = [
        "US_employment_data.csv",
        "germany_population_2023.xlsx", 
        "data_FR_economic.csv",
        "UK-census-data.json",
        "netherlands_survey.csv",
        "random_data_file.csv"
    ]
    
    for filename in test_filenames:
        country_info = country_extractor.process_filename_country(filename)
        print(f"  {filename:<25} -> {country_info['standardized_name']} ({country_info['alpha_2']})")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Create sample data and process it
    print("2. Processing sample data:")
    
    # Create a sample CSV-like structure
    sample_data = {
        "sample_US_data": {
            "metadata": {},
            "columns": {
                "name": {"records": ["John", "Jane", "Bob", "Alice"]},
                "age": {"records": [25, 30, 35, 28]},
                "city": {"records": ["New York", "Los Angeles", "Chicago", "Houston"]},
                "salary": {"records": [50000, 60000, 70000, 55000]}
            }
        }
    }
    
    # Enhance with country metadata
    enhanced_data = {}
    for table_name, table_data in sample_data.items():
        # Simulate file path
        fake_path = Path(f"{table_name}.csv")
        enhanced_table = loader._enhance_table_metadata(table_data, table_name, fake_path)
        enhanced_data[table_name] = enhanced_table
    
    # Display results
    for table_name, table_data in enhanced_data.items():
        metadata = table_data['metadata']
        print(f"Table: {table_name}")
        print(f"  Country: {metadata['country']}")
        print(f"  Country Info: {metadata['country_info']['alpha_2']} ({metadata['country_info']['extraction_method']})")
        print(f"  Columns: {metadata['total_columns']}")
        print(f"  Column Names: {', '.join(metadata['column_names'])}")
    
    print("\n" + "="*50 + "\n")
    print("Example complete!")


if __name__ == "__main__":
    main() 