import os

def load_gitignore(base_path):
    """
    Load patterns from .gitignore and return as a list.
    Only supports simple filenames and folder names (no glob/wildcard parsing for simplicity).
    """
    gitignore_path = os.path.join(base_path, ".gitignore")
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns

def should_skip(item, skip_files):
    """
    Check if item should be skipped based on skip_files list.
    Supports exact names and folder/file names from .gitignore.
    """
    return item in skip_files or item.startswith(".")

def get_project_structure(base_path, indent="", skip_files=None):
    """
    Recursively generates the folder and file structure of a directory as text.
    Skips hidden files/folders, files in skip_files, and files listed in .gitignore.
    """
    if skip_files is None:
        skip_files = []

    structure = ""
    items = sorted(os.listdir(base_path))
    visible_items = [item for item in items if not should_skip(item, skip_files)]

    for i, item in enumerate(visible_items):
        path = os.path.join(base_path, item)
        is_last = (i == len(visible_items) - 1)
        branch = "└── " if is_last else "├── "
        structure += indent + branch + item + "\n"
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            structure += get_project_structure(path, indent + extension, skip_files)
    return structure

if __name__ == "__main__":
    base_dir = "."  # project root

    # Load .gitignore patterns
    gitignore_patterns = load_gitignore(base_dir)

    # Custom files to skip
    custom_skip_files = ["generate_file_structure.py", "prompt.txt", "structure.txt"]

    # Combine both
    skip_files = set(gitignore_patterns + custom_skip_files)

    # Generate structure
    structure_text = get_project_structure(base_dir, skip_files=skip_files)

    # Save to file
    with open("structure.txt", "w", encoding="utf-8") as f:
        f.write(structure_text)

    print("✅ Project structure saved to structure.txt (ignoring .gitignore & custom files)")