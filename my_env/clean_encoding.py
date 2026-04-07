import os

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                cleaned = content.encode("ascii", "ignore").decode()

                with open(path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"Cleaned: {path}")

            except Exception as e:
                print(f"Skipped: {path} ({e})")

print("DONE: Removed all non-ASCII characters")