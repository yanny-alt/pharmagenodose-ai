import pandas as pd
from pharmagenodose import PharmaGenoDoseFramework

# Load data
df = pd.read_csv("data/iwpc_data.csv", low_memory=False)

# Initialize framework
framework = PharmaGenoDoseFramework()

# Create features
processed_df = framework.create_advanced_features(df)

# Check VKORC1 encoding
print("\n=== VKORC1 ENCODING CHECK ===")
print(f"Column exists: {'VKORC1_sensitivity' in processed_df.columns}")

if 'VKORC1_sensitivity' in processed_df.columns:
    print(f"Values: {processed_df['VKORC1_sensitivity'].value_counts()}")
    print(f"Missing: {processed_df['VKORC1_sensitivity'].isna().sum()}")
else:
    print("❌ VKORC1_sensitivity was NOT created!")

# Check genetic risk score
print("\n=== GENETIC RISK SCORE CHECK ===")
print(f"Column exists: {'Genetic_risk_score' in processed_df.columns}")

if 'Genetic_risk_score' in processed_df.columns:
    print(f"Mean: {processed_df['Genetic_risk_score'].mean():.2f}")
    print(f"Range: {processed_df['Genetic_risk_score'].min():.2f} - {processed_df['Genetic_risk_score'].max():.2f}")
else:
    print("❌ Genetic_risk_score was NOT created!")