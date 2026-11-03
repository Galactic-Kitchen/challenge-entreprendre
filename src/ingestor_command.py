import os
from pathlib import Path
from typing import Any, Iterator
from ingestion import initialiser_et_rempli_chromadb

from config import Config

PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projets_sources"

FILES_TO_CHECK = ["README.md", "README.txt", "README", "package.json"]
EXTENSIONS_TO_CHECK = [".md", ".txt", ".toml"]


def iter_project_files() -> Iterator[Path]:
    """Yield files located under project folders up to 3 levels deep.

    A file is kept when its basename matches one of FILES_TO_CHECK or when its
    extension is in EXTENSIONS_TO_CHECK.
    """
    for root, dirs, files in os.walk(PROJECTS_DIR):
        current_dir = Path(root)
        depth = len(current_dir.relative_to(PROJECTS_DIR).parts)

        if depth >= 3:
            dirs.clear()  # Stop descending into subdirectories

        for filename in files:
            file_path = current_dir / filename
            normalized_name = filename.lower()
            normalized_suffix = file_path.suffix.lower()

            if normalized_name in {name.lower() for name in FILES_TO_CHECK} or normalized_suffix in {
                ext.lower() for ext in EXTENSIONS_TO_CHECK
            }:
                yield file_path


if __name__ == "__main__":
    projets: list[dict[str, Any]] = []
    for file_path in iter_project_files():
        relative_path = file_path.relative_to(PROJECTS_DIR)
        nom_projet = relative_path.parts[0] if len(relative_path.parts) > 1 else file_path.parent.name

        try:
            contenu = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            contenu = file_path.read_text(encoding="utf-8", errors="replace")

        projets.append(
            {
                "nom_projet": nom_projet,
                "nom_fichier": file_path.name,
                "type_fichier": file_path.suffix.lower().lstrip("."),
                "contenu": contenu,
            }
        )

    initialiser_et_rempli_chromadb(projets)



