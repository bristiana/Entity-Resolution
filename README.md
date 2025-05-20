# Entity Resolution System for Company Data


> ⚠️ Memory Requirement: This solution requires sufficient available memory to process the dataset. While it is optimized to use less than 1GB of RAM (compared to 8.33GB for a naive approach), please ensure your system has at least 2GB of available memory.
> 

## Overview

Entity Resolution (ER) is the process of identifying and linking different representations of the same real-world entity across various data sources. In the context of company data, this means determining when multiple records actually refer to the same company despite variations in how the information is recorded.

This code implements a comprehensive Entity Resolution system designed to identify unique companies and group duplicate records in large datasets. It uses advanced fuzzy matching, blocking, and graph-based clustering techniques to efficiently process data with potential inconsistencies, variations, and duplications.

The implementation follows a structured pipeline approach:

1. Data loading and preprocessing
2. Blocking to reduce comparison space
3. Pair-wise similarity calculation
4. Graph-based clustering
5. Representative record selection
6. Results generation and export

## Quick Results

- **Total Records**: 33,446 company entries
- **Unique Companies**: 6,591 distinct entities identified
- **Duplicate Records**: 26,855 duplicates found
- **Duplication Rate**: 80.29%
- **Processing Time**: ~10 minutes on standard laptop
- **Memory Usage**: <1GB peak (vs. 8.33GB for naive approach)

## Key Features

### Advanced Fuzzy Matching Techniques

The system employs several sophisticated approaches to match company records:

- **Name Similarity**: Uses the Jaro-Winkler algorithm to compare company names, recognizing similar names despite variations in spelling, abbreviations, or formatting.
- **Multi-attribute Weighted Similarity**: Combines similarity scores from different attributes (name, website domain, phone number, location, industry) with appropriate weights to calculate a comprehensive similarity measure.
- **Normalized Comparisons**: All fields are normalized before comparison, removing punctuation, legal entity types, and other non-essential variations.

### Efficient Blocking Strategy

To make the comparison process computationally feasible for large datasets, the system implements a multi-faceted blocking strategy:

- **Name Prefix Blocking**: Groups companies by the first 3 characters of their normalized name
- **Website Domain Blocking**: Groups companies with identical website domains
- **Phone Number Blocking**: Groups companies with identical normalized phone numbers
- **Geographic Blocking**: Groups companies by country and region

This approach dramatically reduces the number of required comparisons from O(n²) to something much more manageable.

### Graph-based Clustering

The system uses graph theory to identify clusters of similar records:

- Each company record is represented as a node in a graph
- Similar pairs of records are connected by edges
- Connected components in the graph form clusters representing the same real-world entity
- This approach elegantly handles transitivity relationships (if A~B and B~C, then A, B, and C are grouped together)

### Intelligent Representative Record Selection

For each identified cluster, the system selects the most complete and reliable record to represent the unique company:

- A weighted completeness score considers the presence of key company attributes
- More important fields (like company name, website, phone) receive higher weights
- More recent records receive bonus points, favoring up-to-date information
- This ensures that the "canonical" record for each company contains the most complete and accurate information

## Technical Implementation Details

### Data Preprocessing

The preprocessing phase includes several crucial transformations:

- **Company Name Normalization**: Removes legal entity types (LLC, Inc, etc.), punctuation, and converts to lowercase
- **Website Domain Extraction**: Extracts standardized domain names from URLs
- **Phone Number Normalization**: Removes non-digit characters
- **Address Normalization**: Standardizes formatting and removes punctuation
- **Industry Fingerprinting**: Creates a standardized representation of industry information

### Similarity Calculation

The system uses different similarity measures for different types of attributes:

- **Text Fields**: Jaro-Winkler similarity for names and other text fields
- **Geographic Coordinates**: Haversine distance formula to accurately measure physical distance
- **Categorical Data**: Jaccard similarity for industry fingerprints

### Optimization Techniques

Several optimizations make the system scalable to large datasets:

- **Multi-level Blocking**: Reduces the comparison space significantly
- **Length-based Filtering**: For large blocks, only compares names with similar lengths
- **Early Exact Matches**: Immediately links records with identical domains or phone numbers
- **Memory-efficient Processing**: Processes data in chunks and uses efficient data structures

## Memory-Efficient Design

The original approach would require creating a full similarity matrix comparing all 33,446 records with each other:

```
n × n matrix = 33,446 × 33,446 = 1,118,637,316 elements
1,118,637,316 elements × 8 bytes (float64) ≈ 8.33 GB

```

Instead, this implementation uses a blocking-based approach that:

1. **Reduces comparison space** by only comparing records within the same block
2. **Stores only similarity values above threshold** instead of the full matrix
3. **Processes blocks incrementally** to manage memory usage

This reduces memory requirements from 8.33 GB to less than 1 GB, making the solution viable on standard hardware.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/entity-resolution.git
cd entity-resolution

```

1. Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

```

## Usage

The Entity Resolution pipeline is encapsulated in the `EntityResolution` class, which provides a straightforward API:

```python
# Initialize with path to JSON file containing company records
er = EntityResolution("entity_resolution.json")

# Run the complete pipeline with customizable similarity threshold
result_df = er.run_entity_resolution(similarity_threshold=0.7)

# Save results to JSON files (both complete and unique-only versions)
er.save_results("entity_resolution_results.json")

```

To run the entity resolution algorithm:

```bash
python entity_resolution.py

```

This will:

1. Load the company data from `entity_resolution.json`
2. Preprocess the data and create blocking structures
3. Find similar company pairs
4. Build clusters of duplicate records
5. Select representative records for each cluster
6. Save the results to `entity_resolution_results.json` and `entity_resolution_results_unique.json`

## Project Structure

```
entity-resolution/
├── entity_resolution.py    # Main implementation
├── input/                  # Data files
│   ├── entity_resolution.json  # Input dataset
├── output/                 # Data files
│   └── entity_resolution_results_unique.json # Deduplicated results
├── docs/                   # Documentation
│   ├── CODE.md             # Code documentation
│   ├── SOLUTION.md         # Solution approach
│   └── CHALLENGES.md       # Challenges and resolutions
├── requirements.txt        # Dependencies
└── README.md               # This file

```

## Documentation

For more detailed information about the implementation, please refer to:

- [CODE.md]: Detailed documentation of each function in the code
- [SOLUTION.md]: In-depth explanation of the solution approach
- [CHALLENGES.md]: Challenges encountered and how they were resolved

## License

This project is licensed under the MIT License .

## Acknowledgments

- Veridion for providing the interesting entity resolution challenge
- NetworkX library for efficient graph-based clustering
- Jellyfish for string similarity calculations
