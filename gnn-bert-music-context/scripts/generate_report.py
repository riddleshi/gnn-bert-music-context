#!/usr/bin/env python3
"""Compile report/final_report.tex, rasterize pages, publish to the lab.

    PYTHONPATH=. python scripts/generate_report.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from src.config import ROOT


def _pdflatex(tex: Path) -> Path:
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(tex.parent),
        tex.name,
    ]
    subprocess.run(cmd, cwd=tex.parent, check=True, capture_output=True)
    subprocess.run(cmd, cwd=tex.parent, check=True, capture_output=True)
    pdf = tex.with_suffix(".pdf")
    if not pdf.exists():
        raise SystemExit("pdflatex did not write a PDF")
    return pdf


def _rasterize(pdf: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("page-*.png"):
        old.unlink()
    import fitz  # pymupdf

    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        pix.save(dest / f"page-{i:02d}.png")
    n = doc.page_count
    doc.close()
    shutil.copy(pdf, dest / "final_report.pdf")
    md = ROOT / "report" / "final_report.md"
    if md.exists():
        shutil.copy(md, dest / "final_report.md")
    return n


def _set_paper_page_count(n: int, rev: str) -> None:
    path = ROOT / "web" / "src" / "components" / "PaperView.tsx"
    text = path.read_text()
    text = re.sub(r"const PAGE_COUNT = \d+", f"const PAGE_COUNT = {n}", text, count=1)
    text = re.sub(r'const PAPER_REV = "[^"]+"', f'const PAPER_REV = "{rev}"', text, count=1)
    text = re.sub(
        r"\d+-page (?:NeurIPS-style report|IEEE conference paper)",
        f"{n}-page IEEE conference paper",
        text,
    )
    path.write_text(text)


def main() -> None:
    tex = ROOT / "report" / "final_report.tex"
    print("compiling", tex.relative_to(ROOT))
    pdf = _pdflatex(tex)
    n = _rasterize(pdf, ROOT / "web" / "public" / "paper")
    _set_paper_page_count(n, rev=f"{n}p-{pdf.stat().st_size}")
    print(f"wrote {pdf.relative_to(ROOT)} ({n} pages)")
    print("lab copy → web/public/paper/")


if __name__ == "__main__":
    main()
