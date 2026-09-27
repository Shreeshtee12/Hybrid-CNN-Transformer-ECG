"""
dedupe_model_files.py
======================
Fixes files where the ENTIRE content was accidentally pasted twice
(resnet1d.py, xresnet1d.py, transformer.py, xlstm.py, model_selector.py).

For each file, splits the content on the repeated header comment line
(e.g. "# models/resnet1d.py") and keeps only the first copy.

Makes a .bak backup of each file before overwriting, so nothing is lost
if something looks wrong.

Usage (run from src/models/torch_models/):
    python dedupe_model_files.py
"""

import os

FILES_TO_DEDUPE = [
    "resnet1d.py",
    "xresnet1d.py",
    "transformer.py",
    "xlstm.py",
    "model_selector.py",
]


def dedupe_file(path: str):
    with open(path, "r") as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    if not lines:
        print(f"[dedupe] {path}: empty file, skipping.")
        return

    header = lines[0]  # e.g. "# models/resnet1d.py\n"
    # Find the SECOND occurrence of the header line (the start of the duplicate)
    occurrences = [i for i, line in enumerate(lines) if line == header]

    if len(occurrences) < 2:
        print(f"[dedupe] {path}: header appears only once, no duplication found. Skipping.")
        return

    second_start = occurrences[1]
    first_copy = "".join(lines[:second_start])

    # Backup before overwriting
    backup_path = path + ".bak"
    with open(backup_path, "w") as f:
        f.write(content)

    with open(path, "w") as f:
        f.write(first_copy)

    print(f"[dedupe] {path}: removed duplicate (kept lines 1-{second_start}, "
          f"backup saved to {backup_path})")


def main():
    for fname in FILES_TO_DEDUPE:
        if os.path.exists(fname):
            dedupe_file(fname)
        else:
            print(f"[dedupe] {fname}: not found, skipping.")


if __name__ == "__main__":
    main()