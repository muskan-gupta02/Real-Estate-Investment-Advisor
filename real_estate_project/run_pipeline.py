"""
run_pipeline.py  — Run everything in one shot
Usage:  python run_pipeline.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, label):
    print(f"\n{'='*55}")
    print(f"  ▶  {label}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌  Step failed: {label}")
        sys.exit(1)
    print(f"✅  Done: {label}")


if __name__ == "__main__":
    run("python generate_data.py",  "Step 1 — Generate synthetic dataset")
    run("python preprocess.py",     "Step 2 — Preprocess & feature engineering")
    run("python eda.py",            "Step 3 — Run EDA (20 plots → eda_plots/)")
    run("python train_models.py",   "Step 4 — Train models & log with MLflow")
    print("\n" + "="*55)
    print("  🚀  Launching Streamlit app ...")
    print("  Open http://localhost:8501 in your browser")
    print("="*55 + "\n")
    os.system("streamlit run app.py")
