import os

folders = [
    "data/raw",
    "data/processed",
    "data/models",
    "collectors",
    "features",
    "model",
    "dashboard",
    "logs",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully:")
for f in folders:
    print(f"  {f}/")