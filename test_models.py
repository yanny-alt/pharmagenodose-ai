import joblib

# Load trained model
model_data = joblib.load("models/production_model.joblib")

print("\n=== BASE MODEL PERFORMANCE ===")
for name, metrics in model_data['training_metrics']['base_models'].items():
    print(f"{name}: R² = {metrics['R2']:.3f}, Accuracy = {metrics['Clinical_Accuracy_20']:.1f}%")

print("\n=== ENSEMBLE PERFORMANCE ===")
ensemble = model_data['training_metrics']['ensemble']
print(f"Ensemble: R² = {ensemble['R2']:.3f}, Accuracy = {ensemble['Clinical_Accuracy_20']:.1f}%")

print("\n=== PROBLEM? ===")
best_base = max(model_data['training_metrics']['base_models'].items(), 
                key=lambda x: x[1]['R2'])
print(f"Best base model: {best_base[0]} with R² = {best_base[1]['R2']:.3f}")
print(f"Ensemble R²: {ensemble['R2']:.3f}")

if ensemble['R2'] < best_base[1]['R2']:
    print("❌ META-MODEL IS MAKING THINGS WORSE!")