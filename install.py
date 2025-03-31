import subprocess
import sys

packages = [
    "mne", "scipy", "numpy", "pandas", "matplotlib",
    "scikit-learn", "torch", "torchvision"
]

for i in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", i])