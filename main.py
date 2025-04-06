import subprocess

scripts = [
    "3label.py",
    "3label_token.py",
    "3label_transformer.py"
]

for script in scripts:
    print(f"\n Running {script}...")
    try:
        result = subprocess.run(["python", script], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f" Error while running {script}:")
        print(e.stdout)
        print(e.stderr)
        break
