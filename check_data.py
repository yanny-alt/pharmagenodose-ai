import pandas as pd

# Load your data
iwpc_file = "data/iwpc_data.csv"
aa_file = "data/african_american_genetics.csv"

print("Loading IWPC data...")
df = pd.read_csv(iwpc_file, low_memory=False)

print(f"\nDataset shape: {df.shape}")
print(f"\nAll columns ({len(df.columns)}):")
print(df.columns.tolist())

print("\n\nSearching for VKORC1-related columns:")
vkorc1_cols = [col for col in df.columns if 'VKORC1' in col or 'vkorc' in col.lower()]
print(vkorc1_cols)

print("\n\nSearching for CYP2C9-related columns:")
cyp_cols = [col for col in df.columns if 'CYP2C9' in col or 'cyp' in col.lower()]
print(cyp_cols)

print("\n\nFirst few rows of genetic columns:")
genetic_cols = vkorc1_cols + cyp_cols
if genetic_cols:
    print(df[genetic_cols].head())