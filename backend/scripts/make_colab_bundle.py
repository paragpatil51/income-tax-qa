"""Pack the backend code and the source PDF into one zip for the Colab/Kaggle indexing notebook.

    python scripts/make_colab_bundle.py          # writes backend/colab_bundle.zip (~7 MB)

Secrets are never included: .env, caches, models and generated data are skipped.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from taxqa.core.config import ROOT, settings

INCLUDE = ["taxqa", "scripts", "requirements.txt", "pyproject.toml", ".env.example"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".models", "data"}


def main() -> int:
    if not settings.pdf_path.exists():
        print(f"PDF not found: {settings.pdf_path}")
        return 1
    out = ROOT / "colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or f.name == ".env" or f.suffix == ".pyc":
                    continue
                z.write(f, Path("backend") / rel)
        z.write(settings.pdf_path, Path("backend") / settings.pdf_path.relative_to(ROOT))
    names = zipfile.ZipFile(out).namelist()
    assert not any(n.endswith("/.env") or n == "backend/.env" for n in names), "refusing to ship .env"
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files). Upload it in the notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
