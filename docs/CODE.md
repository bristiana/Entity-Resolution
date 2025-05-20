
# Code Documentation

This document provides detailed documentation for each function in the memory-efficient entity resolution solution.

## Table of Contents

1. [Class Overview](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#class-overview)
2. [Data Loading and Preprocessing](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#data-loading-and-preprocessing)
3. [Blocking Strategy](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#blocking-strategy)
4. [Similarity Calculation](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#similarity-calculation)
5. [Clustering](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#clustering)
6. [Representative Selection](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#representative-selection)
7. [Output Generation](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#output-generation)
8. [Runner](https://claude.ai/chat/42709044-90c9-4283-b2e5-8c22317d57a3#runner)

## Class Overview

The solution is implemented as a Python class `EntityResolution` that encapsulates the entire entity resolution pipeline.

```python
class EntityResolution:
    def __init__(self, data_path):
        """Initialize the entity resolution processor.

        Parameters:
        -----------
        data_path : str
            Path to the JSON file containing company data
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.clusters = None
        self.result_df = None

```

The class maintains state through the processing pipeline, allowing for incremental building of the solution.

## Data Loading and Preprocessing

### `load_data()`

```python
def load_data(self):
    """Load company data from the JSON file."""
    with open(self.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    self.df = pd.DataFrame(data)
    print(f"Loaded {len(self.df)} company records.")
    return self.df

```

- **Purpose**: Loads company data from a JSON file into a pandas DataFrame
- **Parameters**: None (uses `self.data_path` from initialization)
- **Returns**: DataFrame containing the company records
- **Notes**: Handles UTF-8 encoding to support international company names

### `preprocess_data()`

```python
def preprocess_data(self):
    """Preprocess the data to prepare for matching."""
    print("Preprocessing data...")
    df = self.df.copy()

    # Handle missing values (replace 'None' strings with actual None)
    for col in df.columns:
        df[col] = df[col].replace('None', None)

    # Create normalized company name
    df['normalized_name'] = df['company_name'].apply(self._normalize_company_name)

    # Create name prefix for blocking
    df['name_prefix'] = df['normalized_name'].apply(
        lambda x: x[:3].lower() if x and len(x) >= 3 else '_' if x else '_'
    )

    # Additional preprocessing steps...

    self.processed_df = df
    print("Preprocessing complete.")
    return self.processed_df

```

- **Purpose**: Prepares raw data for matching by normalizing and standardizing fields
- **Key Transformations**:
    - Replaces "None" strings with actual None values
    - Normalizes company names
    - Creates name prefixes for blocking
    - Extracts domains from URLs
    - Normalizes phone numbers
    - Parses commercial names
    - Normalizes addresses
    - Converts coordinates to numeric values
    - Creates industry fingerprints

### Helper Preprocessing Functions

### `_normalize_company_name(self, name)`

- **Purpose**: Normalizes company names by removing legal entity types, punctuation, and standardizing whitespace
- **Process**:
    - Converts to lowercase
    - Removes legal entity types (LLC, Inc, GmbH, etc.)
    - Replaces special characters with spaces
    - Removes standalone digits
    - Standardizes whitespace

### `_extract_domain(self, url)`

- **Purpose**: Extracts domain from URL
- **Process**:
    - Removes protocol (http://, https://)
    - Removes 'www.' prefix
    - Extracts domain before first slash

### `_normalize_phone(self, phone)`

- **Purpose**: Standardizes phone numbers by removing non-digit characters
- **Process**:
    - Removes all non-digit characters using regex

### `_split_commercial_names(self, names)`

- **Purpose**: Splits pipe-separated commercial names into a list
- **Process**:
    - Splits strings by '|' character
    - Returns list of trimmed name variations

### `_normalize_address(self, address)`

- **Purpose**: Normalizes addresses for comparison
- **Process**:
    - Converts to lowercase
    - Removes punctuation
    - Standardizes whitespace

## Blocking Strategy

### `blocking_strategy()`

```python
def blocking_strategy(self):
    """Create blocks of companies for comparison to reduce memory usage."""
    print("Creating blocks for efficient processing...")
    df = self.processed_df

    # Strategy 1: Block by first 3 characters of normalized name
    name_blocks = df.groupby('name_prefix').indices

    # Strategy 2: Block by website domain (if available)
    domain_blocks = {}
    for domain, group in df.groupby('website_domain'):
        if domain is not None and domain != '':
            domain_blocks[domain] = group.index.tolist()

    # Additional blocking strategies...

    return {
        'name_blocks': name_blocks,
        'domain_blocks': domain_blocks,
        'phone_blocks': phone_blocks,
        'location_blocks': location_blocks
    }

```

- **Purpose**: Creates blocks of similar records to reduce comparison space
- **Blocking Strategies**:
    1. **Name Prefix**: Groups by first 3 characters of normalized name
    2. **Website Domain**: Groups by exact domain match
    3. **Phone Number**: Groups by exact phone match
    4. **Location**: Groups by country and region
- **Returns**: Dictionary of blocks for each strategy
- **Complexity Reduction**: From O(n²) to O(b×k²) where b is the number of blocks and k is the average block size

## Similarity Calculation

### `find_similar_pairs(self, similarity_threshold=0.7)`

```python
def find_similar_pairs(self, similarity_threshold=0.7):
    """Find pairs of similar companies using blocking strategy."""
    print(f"Finding similar company pairs with threshold {similarity_threshold}...")

    # Get blocks
    blocks = self.blocking_strategy()
    df = self.processed_df

    # Define feature weights
    weights = {
        'name': 0.35,            # Company name is very important
        'domain': 0.25,          # Website domain is a strong identifier
        'phone': 0.15,           # Phone can be a good identifier but might change
        'location': 0.15,        # Geographic location is important
        'industry': 0.1,         # Industry can help resolve ambiguous cases
    }

    # Process blocks to find similar pairs...

    print(f"Found {len(similar_pairs)} similar pairs.")
    return similar_pairs

```

- **Purpose**: Identifies pairs of companies that exceed the similarity threshold
- **Parameters**:
    - `similarity_threshold`: Minimum similarity score (default: 0.7)
- **Process**:
    1. Retrieves blocks from blocking strategy
    2. Defines attribute weights
    3. Processes each block to find similar pairs
    4. For exact-match blocks (domain, phone), adds pairs directly
    5. For name blocks, calculates multi-attribute similarity
- **Returns**: List of tuples (idx_i, idx_j, similarity) for similar pairs
- **Memory Efficiency**: Only stores pairs above threshold, not full matrix

### `_calculate_pair_similarity(self, idx_i, idx_j, df, weights)`

```python
def _calculate_pair_similarity(self, idx_i, idx_j, df, weights):
    """Calculate similarity between a pair of company records."""
    # Initialize similarity
    total_similarity = 0

    # Get records
    record_i = df.loc[idx_i]
    record_j = df.loc[idx_j]

    # Calculate similarity for different attributes...

    return total_similarity

```

- **Purpose**: Calculates weighted similarity between two company records
- **Parameters**:
    - `idx_i`, `idx_j`: Indices of the two records
    - `df`: DataFrame containing the records
    - `weights`: Dictionary of attribute weights
- **Attributes Compared**:
    1. **Name similarity**: Using Jaro-Winkler distance for fuzzy matching
    2. **Domain similarity**: Exact matching for high confidence
    3. **Phone similarity**: Exact matching after normalization
    4. **Location similarity**: Geographic proximity or address text similarity
    5. **Industry similarity**: Jaccard similarity between industry fingerprints
- **Returns**: Weighted similarity score between 0 and 1

### `_calculate_name_similarity(self, name1, comm_names1, name2, comm_names2)`

- **Purpose**: Calculates the maximum similarity between company names and commercial names
- **Process**:
    - Compares primary names using Jaro-Winkler similarity
    - Compares primary name with commercial names
    - Compares commercial names with each other
    - Returns maximum similarity found

### `_haversine_distance(self, lat1, lon1, lat2, lon2)`

- **Purpose**: Calculates geographic distance between two locations
- **Formula**: Uses Haversine formula to calculate distance on a sphere
- **Returns**: Distance in kilometers

## Clustering

### `build_clusters_from_pairs(self, similar_pairs)`

```python
def build_clusters_from_pairs(self, similar_pairs):
    """Build clusters from similar pairs using graph-based clustering."""
    print("Building clusters from similar pairs...")

    # Create a graph where nodes are company records
    G = nx.Graph()

    # Add all nodes (company records)
    for i in range(len(self.processed_df)):
        G.add_node(i)

    # Add edges between similar companies
    for idx_i, idx_j, similarity in similar_pairs:
        G.add_edge(idx_i, idx_j, weight=similarity)

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Convert to dictionary and handle singletons...

    print(f"Identified {next_cluster_id} clusters")
    return self.processed_df

```

- **Purpose**: Groups similar company records into clusters
- **Algorithm**: Graph-based clustering using NetworkX
- **Process**:
    1. Creates a graph with nodes as company records
    2. Adds edges between similar pairs with similarity as weight
    3. Identifies connected components (subgraphs) as clusters
    4. Converts clusters to a dictionary mapping record index to cluster ID
    5. Handles singleton records (records without connections)
- **Returns**: DataFrame with cluster ID assigned to each record
- **Advantage**: Handles transitive relationships (if A matches B and B matches C, then A, B, and C are in the same cluster)

## Representative Selection

### `select_representative_records(self)`

```python
def select_representative_records(self):
    """Select the most representative record for each cluster."""
    print("Selecting representative records for each cluster...")

    df = self.processed_df
    result_df = df.copy()

    # Count clusters
    clusters = df['cluster_id'].unique()
    print(f"Processing {len(clusters)} clusters")

    # For each cluster, select the most complete and reliable record...

    self.result_df = result_df
    print("Representative record selection complete.")
    return result_df

```

- **Purpose**: Selects the most complete and reliable record for each cluster
- **Process**:
    1. For each cluster, examines all records
    2. Calculates completeness score based on presence of key fields
    3. Weights earlier fields (name, website, phone) more heavily
    4. Adds bonus for newer records (based on last_updated_at)
    5. Selects record with highest score as representative
- **Returns**: DataFrame with representative status for each record
- **Field Importance**: Company name > website domain > phone > address > coordinates > descriptions > industry codes

## Output Generation

### `create_output_dataset(self)`

```python
def create_output_dataset(self):
    """Create the final output dataset with cluster information."""
    print("Creating final output dataset...")

    output_df = self.result_df.copy()

    # Calculate cluster sizes
    cluster_sizes = output_df['cluster_id'].value_counts().to_dict()
    output_df['cluster_size'] = output_df['cluster_id'].map(cluster_sizes)

    # Add duplicate IDs within each cluster...

    print("Final output dataset ready.")
    return output_df

```

- **Purpose**: Creates the final output dataset with all cluster information
- **Process**:
    1. Calculates cluster sizes
    2. Adds duplicate IDs to each representative record
    3. Sorts records by cluster ID and representative status
- **Returns**: Complete output dataset ready for saving

### `save_results(self, output_path)`

```python
def save_results(self, output_path):
    """Save the results to a JSON file."""
    if self.result_df is None:
        print("No results to save. Run entity resolution first.")
        return

    # Create a simplified output with just the essential columns...

    # Convert to json
    result_json = output_df.to_json(orient='records')

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result_json)

    print(f"Results saved to {output_path}")

    # Save only unique records (representatives) if needed...

```

- **Purpose**: Saves results to JSON files
- **Outputs**:
    1. Complete results with all records and cluster information
    2. Deduplicated results with only representative records
- **Format**: JSON records format, with UTF-8 encoding

## Runner

### `run_entity_resolution(self, similarity_threshold=0.7)`

```python
def run_entity_resolution(self, similarity_threshold=0.7):
    """Run the complete entity resolution pipeline."""
    self.load_data()
    self.preprocess_data()
    similar_pairs = self.find_similar_pairs(similarity_threshold)
    self.build_clusters_from_pairs(similar_pairs)
    self.select_representative_records()
    return self.create_output_dataset()

```

- **Purpose**: Executes the complete entity resolution pipeline
- **Parameters**:
    - `similarity_threshold`: Minimum similarity for company matching (default: 0.7)
- **Process**:
    1. Loads data from JSON file
    2. Preprocesses data for matching
    3. Finds similar company pairs
    4. Builds clusters from similar pairs
    5. Selects representative records
    6. Creates output dataset
- **Returns**: Final output dataset with all cluster information
- **Usage**: Main entry point for running the entity resolution process
