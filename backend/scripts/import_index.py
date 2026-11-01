"""Install an index built elsewhere (e.g. the Colab notebook) into backend/data.

    python scripts/import_index.py path/to/index_bundle.zip

Only data/processed/* and data/index/* are extracted. The bundle is checked
against this machine first: it must have been built from the same PDF (SHA-256)
and with the embedding model configured in .env, otherwise the app would refuse
to load it or, worse, search with mismatched vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from taxqa.core.config import ROOT, settings
from taxqa.retrieval.embeddings import Embedder

ALLOWED = ("data/processed/", "data/index/")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--force", action="store_true", help="skip the PDF hash check")
    args = ap.parse_args()

    with zipfile.ZipFile(args.bundle) as z:
        names = [n for n in z.namelist() if n.startswith(ALLOWED) and not n.endswith("/")]
        if not any(n.endswith("index/meta.json") for n in names):
            print("Not an index bundle (no data/index/meta.json).")
            return 1
        meta = json.loads(z.read("data/index/meta.json"))
        report = json.loads(z.read("data/processed/ingest_report.json"))

        embedder = Embedder(settings)
        if (meta["backend"], meta["model"]) != (embedder.backend, embedder.name):
            print(f"Bundle embeddings are {meta['backend']}:{meta['model']}, but .env configures "
                  f"{embedder.backend}:{embedder.name}. Align EMBEDDING_BACKEND / LOCAL_EMBEDDING_MODEL first.")
            return 1
        if settings.pdf_path.exists() and not args.force:
            local = _sha256(settings.pdf_path)
            if local != report.get("source_sha256"):
                print("The bundle was built from a different PDF than backend/data/raw (SHA-256 differs). "
                      "Use --force to import anyway.")
                return 1

        for n in names:
            target = ROOT / n
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))

    print(f"Imported {len(names)} files: {meta['count']} chunks, embeddings {meta['backend']}:{meta['model']}, "
          f"built {meta['built_at']}.")
    print(f"Source: {report.get('source_pdf')} | sections {report.get('sections_found')} | "
          f"chapters {len(report.get('chapters_found', []))}")
    print("Start the app with:  uvicorn taxqa.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
