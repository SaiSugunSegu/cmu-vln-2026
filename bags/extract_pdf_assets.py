#!/usr/bin/env python3
"""Extract the per-scene challenge question PDFs into reviewable image assets.

``questions/<scene>/questions.pdf`` holds the five official questions for a scene,
each next to a screenshot of the situation it asks about. Those screenshots are the
only visual reference for authoring category-2 (object reference) questions, so this
dumps them as PNGs plus a JSON index tying each question to the images on its page.

Output (untracked, see .gitignore)::

    data/pdf_assets/<scene>/page1.png          150-dpi render of the whole page
    data/pdf_assets/<scene>/p1_img1.png        embedded screenshot, page 1, first image
    data/pdf_assets/<scene>/pdf_text.json      {pages: [...], questions: [...]}

Usage::

    python3 bags/extract_pdf_assets.py                 # every scene under questions/
    python3 bags/extract_pdf_assets.py arabic_room
    python3 bags/extract_pdf_assets.py --dpi 200 --force
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

REPO = Path(__file__).resolve().parent.parent
QUESTIONS_ROOT = REPO / "questions"
DEFAULT_OUT_ROOT = REPO / "data" / "pdf_assets"

SCENE_NAME_RE = re.compile(r"^[a-z0-9_]+$")
CATEGORY_RE = re.compile(r"^Category\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
QUESTION_RE = re.compile(r"^Question\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
RESPONSE_RE = re.compile(r"^Response\s*:\s*(.*)$", re.IGNORECASE)
PAGE_FOOTER_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$")

HEADING_TOL = 24.0  # points; roughly two text lines

CATEGORY_SLUG = {
    1: "numerical",
    2: "object_reference",
    3: "instruction_following",
}


def scene_pdf(scene: str) -> Path:
    """Resolve a scene's questions.pdf, refusing anything outside questions/."""
    if not SCENE_NAME_RE.match(scene):
        raise ValueError(f"invalid scene name: {scene!r}")
    path = (QUESTIONS_ROOT / scene / "questions.pdf").resolve()
    if not str(path).startswith(str(QUESTIONS_ROOT.resolve())):
        raise ValueError(f"scene path escapes {QUESTIONS_ROOT}: {scene!r}")
    return path


def discover_scenes() -> list[str]:
    if not QUESTIONS_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in QUESTIONS_ROOT.iterdir()
        if p.is_dir() and SCENE_NAME_RE.match(p.name) and (p / "questions.pdf").exists()
    )


def question_images(page: fitz.Page, names: dict[int, str]) -> dict[int, list[str]]:
    """Assign each screenshot on a page to the question it illustrates.

    A page carries two questions and two screenshots, so a page-level pairing would
    hand both images to both questions. The layout is strictly top-to-bottom, so an
    image belongs to the last ``Question N:`` heading above it.
    """
    headings: list[tuple[float, int]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line.get("spans", []))
            if m := QUESTION_RE.match(text.strip()):
                headings.append((line["bbox"][1], int(m.group(1))))
    headings.sort()

    out: dict[int, list[str]] = {}
    for info in page.get_image_info(xrefs=True):
        name = names.get(info.get("xref", -1))
        if not name:
            continue
        top = info["bbox"][1]
        # A screenshot can start a couple of points above the baseline of the heading
        # it belongs to, so a strict "heading above image" test misattributes it.
        owners = [num for y, num in headings if y <= top + HEADING_TOL]
        # No heading above it: the image continues the question the page opened with,
        # which parse_questions resolves from the running state (key 0).
        out.setdefault(owners[-1] if owners else 0, []).append(name)
    return out


def parse_questions(pages: list[dict]) -> list[dict]:
    """Walk the page texts and recover (category, question, response, page) tuples.

    A question can wrap across lines and even across a page break, so lines are
    accumulated until the next ``Response:``/``Question N:``/``Category N:`` marker.
    """
    questions: list[dict] = []
    category: tuple[int, str] | None = None
    current: dict | None = None
    field = None  # which multi-line field the following lines belong to

    def close() -> None:
        nonlocal current, field
        if current is not None:
            current["question"] = " ".join(current["question"].split())
            current["response"] = " ".join(current["response"].split())
            questions.append(current)
        current = None
        field = None

    for page in pages:
        owned = {int(k): v for k, v in page["question_images"].items()}
        if current is not None:
            current["images"].extend(owned.get(0, []))
        for raw_line in page["text"].splitlines():
            line = raw_line.strip()
            if not line or PAGE_FOOTER_RE.match(line):
                continue
            if m := CATEGORY_RE.match(line):
                close()
                category = (int(m.group(1)), m.group(2).strip())
                continue
            if m := QUESTION_RE.match(line):
                close()
                cat_id = category[0] if category else 0
                number = int(m.group(1))
                current = {
                    "number": number,
                    "category": cat_id,
                    "category_name": CATEGORY_SLUG.get(cat_id, category[1].lower() if category else ""),
                    "question": m.group(2),
                    "response": "",
                    "pages": [page["page"]],
                    "images": list(owned.get(number, [])),
                    "page_images": list(page["images"]),
                }
                field = "question"
                continue
            if m := RESPONSE_RE.match(line):
                if current is not None:
                    current["response"] = m.group(1)
                    field = "response"
                continue
            if current is not None and field:
                current[field] += " " + line
                if page["page"] not in current["pages"]:
                    current["pages"].append(page["page"])
                    current["page_images"].extend(page["images"])
    close()
    return questions


def extract_scene(scene: str, out_root: Path, dpi: int, force: bool) -> dict:
    pdf_path = scene_pdf(scene)
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    seen_xrefs: dict[int, str] = {}
    try:
        for index, page in enumerate(doc, start=1):
            page_png = out_dir / f"page{index}.png"
            if force or not page_png.exists():
                page.get_pixmap(dpi=dpi).save(page_png)

            images: list[str] = []
            for slot, info in enumerate(page.get_images(full=True), start=1):
                xref = info[0]
                if xref in seen_xrefs:
                    images.append(seen_xrefs[xref])
                    continue
                raw = doc.extract_image(xref)
                name = f"p{index}_img{slot}.{raw['ext']}"
                target = out_dir / name
                if force or not target.exists():
                    target.write_bytes(raw["image"])
                seen_xrefs[xref] = name
                images.append(name)

            pages.append(
                {
                    "page": index,
                    "render": page_png.name,
                    "images": images,
                    "question_images": question_images(page, seen_xrefs),
                    "text": page.get_text(),
                }
            )
    finally:
        doc.close()

    index_data = {
        "scene": scene,
        "source_pdf": str(pdf_path.relative_to(REPO)),
        "dpi": dpi,
        "pages": pages,
        "questions": parse_questions(pages),
    }
    (out_dir / "pdf_text.json").write_text(json.dumps(index_data, indent=2) + "\n")
    return index_data


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenes", nargs="*", help="Scene names (default: every scene under questions/)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root")
    ap.add_argument("--dpi", type=int, default=150, help="Page render resolution")
    ap.add_argument("--force", action="store_true", help="Rewrite images that already exist")
    args = ap.parse_args()

    scenes = args.scenes or discover_scenes()
    if not scenes:
        print(f"no questions.pdf found under {QUESTIONS_ROOT}")
        return 1

    for scene in scenes:
        data = extract_scene(scene, args.out_root, args.dpi, args.force)
        n_imgs = sum(len(p["images"]) for p in data["pages"])
        cat2 = [q for q in data["questions"] if q["category"] == 2]
        print(
            f"{scene:16s} pages={len(data['pages']):2d} images={n_imgs:2d} "
            f"questions={len(data['questions'])} (category-2: {len(cat2)})"
        )
    print(f"\nwrote {len(scenes)} scene(s) to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
