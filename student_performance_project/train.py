"""Run reproducible training and print the measured model results."""
from src.pipeline import run_training

def main():
    data, result = run_training()
    metrics, best_name, X, y, X_test, y_test, model = result
    print(f"Rows: {len(data)} | Features used: {X.shape[1]} | Target: G3")
    print("G1 and G2 are excluded from model inputs to represent earlier prediction.")
    print("\nMeasured results (not pre-filled or fabricated):")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSelected by lowest mean 5-fold CV MAE: {best_name}")
    print("Artifacts saved in ./artifacts/")

if __name__ == "__main__":
    main()
