# Environment verification script for Python
import sys

print("=== Python Environment Check ===")
print(f"Python Version: {sys.version}")
print(f"Executable:     {sys.executable}\n")

packages = [
    ("numpy", lambda: __import__("numpy").__version__),
    ("pandas", lambda: __import__("pandas").__version__),
    ("scikit-learn", lambda: __import__("sklearn").__version__),
    ("torch", lambda: __import__("torch").__version__),
    ("matplotlib", lambda: __import__("matplotlib").__version__),
    ("seaborn", lambda: __import__("seaborn").__version__),
    ("shap", lambda: __import__("shap").__version__),
    ("jupyter", lambda: __import__("jupyter_core").__version__),
]

print("Importing packages and checking versions:")
for name, get_version in packages:
    try:
        ver = get_version()
        print(f" - {name:<14}: {ver}")
    except Exception as e:
        print(f" - {name:<14}: FAILED ({e})")

print("\nAll Python packages imported successfully.")
