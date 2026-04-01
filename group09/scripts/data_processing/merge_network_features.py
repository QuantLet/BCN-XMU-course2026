import pandas as pd

# Load both CSV files
base_df = pd.read_csv('data/processed/txs_merged.csv')
network_df = pd.read_csv('data/processed/txs_network_features.csv')

# Perform left merge on txId column
merged_df = base_df.merge(network_df, on='txId', how='left')

# Fill NaN values with 0 (isolated nodes not in edgelist)
merged_df.fillna(0, inplace=True)

# Export to new file without index
merged_df.to_csv('data/processed/txs_merge_with_network.csv', index=False)

# Print shape to confirm
print(f"Final dataset shape: {merged_df.shape}")
