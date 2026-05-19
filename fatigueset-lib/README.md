# FatigueSet Library

FatigueSet is a Python library designed for loading, processing, and analyzing multimodal datasets related to physical and mental fatigue. This library provides a set of tools to facilitate data handling, feature extraction, and validation, making it easier for researchers and practitioners to work with physiological and cognitive data.

## Features

- **Data Loading**: Easily load data from CSV and other formats using the `DataLoader` class.
- **Data Processing**: Process and manipulate datasets with the `DataProcessor` class, including filtering and aggregation methods.
- **Feature Extraction**: Extract relevant features and compute statistics using the `FeatureExtractor` class.
- **Data Validation**: Ensure data integrity and quality with the `DataValidator` class, which checks for schema compliance and missing values.
- **Utility Functions**: Access various utility functions for data manipulation and transformation.

## Installation

To install the FatigueSet library, you can use pip. First, clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd fatigueset-lib
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Here is a simple example of how to use the FatigueSet library:

```python
from fatigueset.loader import DataLoader
from fatigueset.processor import DataProcessor

# Load data
loader = DataLoader()
data = loader.load_csv('path/to/data.csv')

# Process data
processor = DataProcessor()
processed_data = processor.process_data(data)
```

For more detailed examples, please refer to the Jupyter notebook located in the `notebooks` directory.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.