"""Render docs/RESULTS.md as a Word document in IEEE format.

Reads the narrative, results/tables/manifest.csv for table numbers, titles and
notes, the table bodies, and the 600-dpi figures. Nothing is retyped.

IEEE conventions applied: table caption above in small capitals, figure caption
below, horizontal rules only, 8 pt table text, figures at the 3.5 in
single-column measure. Tables and figures are placed at the end of the section
that first cites them; anything uncited goes to an appendix.

    python tools/build_results_docx.py . docs/Results.docx
"""

from __future__ import annotations

import csv
import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2
                      else os.path.join(REPO, "docs", "Results.docx"))
MD = os.path.join(REPO, "docs", "RESULTS.md")
TBL = os.path.join(REPO, "results", "tables")
FIG = os.path.join(REPO, "results", "figures", "manuscript")

BODY_PT = 10.0
SMALL_PT = 8.0
FONT = "Times New Roman"
IEEE_COL = 3.5          # inches, single-column measure

# Figures are numbered in the narrative, so the mapping is explicit.
FIGURES = {
    1: ("fig1_approach_comparison",
        "Waste-class F1 on the spatially clean held-out partition for the three "
        "mapping approaches. Points are means over three seeds; bars span one "
        "standard deviation. The pretrained model is the one selected on the "
        "validation split."),
    2: ("fig2_label_budget",
        "Waste-class F1 against the number of annotated training tiles, by "
        "pretraining corpus. The dotted line marks the accuracy of the scratch "
        "model trained on the full budget of 639 tiles; pretrained encoders "
        "reach it with 64."),
    3: ("fig3_frozen_vs_finetuned",
        "Frozen against fine-tuned encoders at the full annotation budget. "
        "Arrows run from the frozen to the fine-tuned result for each backbone; "
        "the direction is consistent across all five."),
    4: ("fig4_prevalence_precision",
        "Precision implied by the Bayes projection as a function of prevalence, "
        "for the selected model and for the configuration with the lowest gap "
        "at recall above 0.70. Vertical rules mark the prevalence of the "
        "annotated tiles and of the landscape. Recall is invariant to "
        "prevalence; precision is not."),
    5: ("fig5_operating_frontier",
        "The attainable operating set. Each open circle is one of the 54 "
        "configurations; filled circles joined by a line are the non-dominated "
        "points, for which no other configuration attains both a higher recall "
        "and a lower deployability gap. The horizontal rule at 1.0 is the value "
        "a map usable without verification would require. The vertical axis is "
        "logarithmic."),
    6: ("fig6_landcover_confusion",
        "Confusability against extent, for the eight non-waste land-cover "
        "classes. The vertical axis is the fraction of a class predicted waste "
        "and the horizontal axis is the share of the scene the class occupies, "
        "both logarithmic; point area is the class's share of all false "
        "positives, and vertical bars span the range across configurations. "
        "Classes at the upper left are intrinsically confusable but rare; those "
        "at the lower right are barely confusable but dominate the error "
        "budget by extent."),
    7: ("fig7_qualitative_landcover",
        "Predictions on four held-out tiles, one for each land-cover class of "
        "Table VIII. Each tile is the one carrying the most false-positive "
        "pixels of that class, so the examples are selected by the measurement "
        "rather than by eye. The reference column shows the digitised waste "
        "polygons; the prediction column separates the three outcomes."),
    8: ("fig8_resolution_robustness",
        "Waste-class F1 against effective ground sample distance. The held-out "
        "imagery was reduced by area averaging and restored to the evaluation "
        "grid, so the reference mask is unchanged throughout. Six "
        "configurations are shown; Table IX gives all twelve. The horizontal "
        "axis is logarithmic."),
    9: ("fig9_qualitative_resolution",
        "The same held-out tile at three effective ground sample distances, "
        "predicted by the model trained from scratch and by the DINOv2 "
        "encoder. Outcome colours as in Fig. 7."),
    10: ("fig10_brightness_robustness",
        "Waste-class F1 against a multiplicative brightness gain applied to "
        "the held-out imagery. A gain of 1.00 is the unperturbed image. Same "
        "six configurations as Fig. 8; Table X gives all twelve."),
}

def roman_value(s: str) -> int:
    """Roman numeral to integer, so table order never depends on a lookup
    table that has to be extended every time a table is added."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s.upper()):
        v = vals[ch]
        total += -v if i + 1 < len(s) and vals[s[i + 1].upper()] > v else v
    return total


class _Roman(dict):
    def __missing__(self, k):
        return roman_value(k)


ROMAN = _Roman()


# --------------------------------------------------------------- document set-up
def new_document() -> Document:
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.0)
        s.left_margin = s.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for name, size in (("Heading 1", 11.5), ("Heading 2", 10.5), ("Heading 3", 10.0)):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.italic = name == "Heading 3"
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(4)
    return doc


def rule(cell, edge: str, size: int) -> None:
    """Add a single horizontal rule to one edge of a cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    el = OxmlElement("w:" + edge)
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:color"), "000000")
    borders.append(el)


def runs(p, text: str, size: float = BODY_PT, italic: bool = False):
    """Render **bold**, *italic* and `code` inline markup."""
    for part in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", text):
        if not part:
            continue
        bold = code = False
        it = italic
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*"):
            part, it = part[1:-1], True
        elif part.startswith("`") and part.endswith("`"):
            part, code = part[1:-1], True
        r = p.add_run(part)
        r.bold, r.italic = bold, it
        r.font.name = "Consolas" if code else FONT
        r.font.size = Pt(size - 1 if code else size)
    return p


# ------------------------------------------------------------------ table output
def load_manifest() -> dict:
    out = {}
    with open(os.path.join(TBL, "manifest.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["roman"]] = row
    return out


def add_table(doc: Document, entry: dict) -> None:
    path = os.path.join(TBL, entry["file"])
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    ncol = len(header)
    size = SMALL_PT if ncol >= 9 else SMALL_PT + 0.5

    # caption, above the table, IEEE style
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(12)
    cap.paragraph_format.space_after = Pt(0)
    cap.paragraph_format.line_spacing = 1.0
    r = cap.add_run("TABLE " + entry["roman"])
    r.font.name, r.font.size = FONT, Pt(SMALL_PT)

    ttl = doc.add_paragraph()
    ttl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ttl.paragraph_format.space_before = Pt(0)
    ttl.paragraph_format.space_after = Pt(3)
    ttl.paragraph_format.line_spacing = 1.0
    r = ttl.add_run(entry["title"])
    r.font.name, r.font.size = FONT, Pt(SMALL_PT)
    r.font.small_caps = True

    t = doc.add_table(rows=1, cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"          # replaced below by horizontal rules only
    clear_all_borders(t)

    numeric = [
        all(re.fullmatch(r"[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?", (r[i] or "").strip())
            or (r[i] or "").strip() in ("--", "n.d.", "yes", "no")
            for r in body)
        for i in range(ncol)
    ]

    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        if numeric[i]:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name, r.font.size, r.bold = FONT, Pt(size), True
        rule(cell, "top", 12)
        rule(cell, "bottom", 6)

    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            if numeric[i]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(v)
            r.font.name, r.font.size = FONT, Pt(size)

    for cell in t.rows[-1].cells:
        rule(cell, "bottom", 12)

    if entry.get("note"):
        n = doc.add_paragraph()
        n.paragraph_format.space_before = Pt(3)
        n.paragraph_format.space_after = Pt(10)
        n.paragraph_format.line_spacing = 1.0
        r = n.add_run(entry["note"])
        r.font.name, r.font.size, r.italic = FONT, Pt(SMALL_PT - 0.5), True
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(8)


def clear_all_borders(table) -> None:
    tbl = table._tbl
    pr = tbl.tblPr
    for old in pr.findall(qn("w:tblBorders")):
        pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    pr.append(borders)


def add_figure(doc: Document, number: int) -> bool:
    stem, caption = FIGURES[number]
    path = os.path.join(FIG, stem + ".png")
    if not os.path.exists(path):
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(path, width=Inches(IEEE_COL))

    c = doc.add_paragraph()
    c.paragraph_format.space_after = Pt(12)
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run("Fig. %d.  " % number)
    r.font.name, r.font.size = FONT, Pt(SMALL_PT)
    r = c.add_run(caption)
    r.font.name, r.font.size = FONT, Pt(SMALL_PT)
    return True


# ----------------------------------------------------------------------- narrative
def build() -> None:
    manifest = load_manifest()
    doc = new_document()

    title = doc.add_paragraph()
    r = title.add_run("Mapping Unmanaged Waste in an Informal Urban Settlement")
    r.bold, r.font.size, r.font.name = True, Pt(15), FONT
    sub = doc.add_paragraph()
    r = sub.add_run("Results")
    r.italic, r.font.size, r.font.name = True, Pt(11), FONT
    r = sub.add_run(
        "\nGenerated from results/ in the project repository. Tables are produced "
        "by 06_evaluation/06_build_manuscript_tables.R and "
        "06_evaluation/07_error_location.R, figures by "
        "07_figures/10_manuscript_figures.R. Formatted to IEEE conventions: "
        "table captions above, figure captions below, figures at the 3.5 in "
        "single-column measure.")
    r.font.size, r.font.name = Pt(SMALL_PT), FONT
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    with open(MD, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    pending_tables: list[str] = []
    pending_figs: list[int] = []
    placed_tables: set[str] = set()
    placed_figs: set[int] = set()
    buf: list[str] = []

    def flush_pending() -> None:
        while pending_tables:
            add_table(doc, manifest[pending_tables.pop(0)])
        while pending_figs:
            add_figure(doc, pending_figs.pop(0))

    def flush_paragraph() -> None:
        nonlocal buf
        text = " ".join(x.strip() for x in buf).strip()
        buf = []
        if not text:
            return
        for rm in re.findall(r"Table\s+([IVX]+)\b", text):
            if rm in manifest and rm not in placed_tables:
                pending_tables.append(rm)
                placed_tables.add(rm)
        for n in re.findall(r"Fig\.\s*(\d+)", text):
            k = int(n)
            if k in FIGURES and k not in placed_figs:
                pending_figs.append(k)
                placed_figs.add(k)
        if re.fullmatch(r"\*\*\[[^\]]+\]\*\*", text):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(text.strip("*"))
            r.italic, r.font.size, r.font.name = True, Pt(SMALL_PT), FONT
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            return
        runs(doc.add_paragraph(), text)

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("# "):
            flush_paragraph(); flush_pending(); doc.add_heading(s[2:], level=1)
        elif s.startswith("## "):
            flush_paragraph(); flush_pending(); doc.add_heading(s[3:], level=2)
        elif s.startswith("### "):
            flush_paragraph(); flush_pending(); doc.add_heading(s[4:], level=3)
        elif s == "---" or not s:
            flush_paragraph()
        elif re.match(r"^\d+\.\s", s):
            flush_paragraph()
            item = [s]
            while i + 1 < len(lines) and lines[i + 1].startswith("   ") and lines[i + 1].strip():
                i += 1
                item.append(lines[i].strip())
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.line_spacing = 1.25
            runs(p, re.sub(r"^\d+\.\s*", "", " ".join(item)))
        elif line.startswith("    "):
            flush_paragraph()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(s)
            r.font.name, r.italic, r.font.size = "Cambria Math", True, Pt(BODY_PT + 0.5)
        else:
            buf.append(line)
        i += 1
    flush_paragraph()
    flush_pending()

    left = [rm for rm in sorted(manifest, key=lambda k: ROMAN[k])
            if rm not in placed_tables]
    if left:
        doc.add_page_break()
        doc.add_heading("Appendix", level=1)
        for rm in left:
            add_table(doc, manifest[rm])

    doc.save(OUT)
    print("wrote", OUT)
    print("tables placed:", ", ".join(sorted(placed_tables, key=lambda k: ROMAN[k])))
    print("figures placed:", ", ".join(str(n) for n in sorted(placed_figs)))
    if left:
        print("in appendix   :", ", ".join(left))


if __name__ == "__main__":
    build()
