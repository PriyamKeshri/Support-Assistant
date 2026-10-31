"""
Renders the Markdown policies in data/policies_src/ into the PDFs the assistant ingests.

The Markdown files are the editable source; the PDFs are what a real company would hand over.
Each PDF gets a running footer, so ingestion has to cope with page furniture like a real document.

    python scripts/build_policy_pdfs.py
"""
import re
from html import escape
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "policies_src"
OUT = ROOT / "data" / "policies"

GREY = HexColor("#555555")
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4)
META = ParagraphStyle("meta", fontName="Helvetica", fontSize=8.5, leading=11, textColor=GREY, spaceAfter=10)
HEADING = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                         spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13.5, spaceAfter=5)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)


def parse_markdown(text: str) -> tuple[str, str, list]:
    """Returns (title, meta line, [(kind, text)]) for the small Markdown subset the policies use."""
    title, meta, blocks, para = "", "", [], []

    def flush():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("# "):
            title = line[2:]
        elif line.startswith("Policy ID:") and not blocks:
            meta = line
        elif line.startswith("## "):
            flush()
            blocks.append(("h", line[3:]))
        elif line.startswith("- "):
            flush()
            blocks.append(("li", line[2:]))
        else:
            para.append(line)
    flush()
    return title, meta, blocks


def build(md_path: Path) -> Path:
    title, meta, blocks = parse_markdown(md_path.read_text())
    code = re.search(r"Policy ID:\s*(\S+)", meta).group(1)
    out = OUT / (md_path.stem + ".pdf")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(20 * mm, 12 * mm, f"Kestrel Analytics | Internal | {code} | Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(escape(title), TITLE), Paragraph(escape(meta), META)]
    for kind, text in blocks:
        if kind == "h":
            story.append(Paragraph(escape(text), HEADING))
        elif kind == "li":
            story.append(Paragraph(escape(text), BULLET, bulletText="•"))
        else:
            story.append(Paragraph(escape(text), BODY))
    story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=22 * mm,
                            title=title, author="Kestrel Analytics IT", subject=code)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for md_path in sorted(SRC.glob("*.md")):
        print(f"wrote {build(md_path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
