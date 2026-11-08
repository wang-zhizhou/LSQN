"""Render the archived macOS and Linux runtime tables, without running solvers.

Requires reportlab >= 4, matplotlib >= 3.9, and Poppler's pdftoppm.
Run from any working directory:
    python scripts/plot_runtime_tables.py
PDF/PNG/LaTeX full tables and per-family PNGs go to benchmarks/figures/.
The numerical inputs are the released archives; no external paths are used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile

from summarize_benchmarks import FAMILIES, METHODS, ranks

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
TITLES = {"macos": "Apple M2 / macOS", "linux": "Intel Core i9-14900K / Linux"}
SUBTITLES = {
    "macos": "Historical paper timings from multiple recorded batches",
    "linux": "Independent release-candidate evaluation on 25 October 2026",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(row):
    return row["family"], row["n"], row["gamma"], row["method"]


def load_rows(platform):
    if platform == "macos":
        paths = [BENCH / "macos-m2-paper/main.json",
                 BENCH / "macos-m2-paper/sources/final_cell_provenance.json"]
        rows = read(paths[0])["records"]
        display = {}
        for r in read(paths[1]):
            r = dict(r, method="LSQN" if r["method"] == "SCB" else r["method"])
            display[cell_key(r)] = r
        for row in rows:
            source = display[cell_key(row)]
            if source["success"] != row["all_success"]:
                raise ValueError("macOS source status mismatch")
            if row["all_success"]:
                if source["seconds"] != row["median_wall_seconds"]:
                    raise ValueError("macOS source time mismatch")
                row["display"] = f'{row["median_wall_seconds"]:.4f}'
                if source["display"] != row["display"]:
                    raise ValueError("macOS paper rounding mismatch")
            else:
                if not re.fullmatch(r"TO|F\([0-3]/3\)", source["display"]):
                    raise ValueError("Unknown macOS failure label")
                row["display"] = source["display"]
    else:
        paths = [BENCH / "linux-intel-2026-10-25/main/analysis.json",
                 BENCH / "linux-intel-2026-10-25/main/summary.json"]
        groups = defaultdict(list)
        for run in read(paths[1])["runs"]:
            groups[(run["case_id"], run["method"])].append(run)
        rows = []
        for entry in read(paths[0])["cases"]:
            runs = groups[(entry["case_id"], entry["method"])]
            if len(runs) != 3 or {r["repeat"] for r in runs} != {0, 1, 2}:
                raise ValueError("Incomplete Linux repetitions")
            successful = sum(r["success"] for r in runs)
            if entry["all_success"] != (successful == 3):
                raise ValueError("Linux summary status mismatch")
            row = dict(entry["spec"], method=entry["method"],
                       all_success=entry["all_success"],
                       median_wall_seconds=entry["median_wall_seconds"])
            if row["all_success"]:
                if statistics.median(r["wall_seconds"] for r in runs) != row["median_wall_seconds"]:
                    raise ValueError("Linux median mismatch")
                row["display"] = f'{row["median_wall_seconds"]:.4f}'
            else:
                row["display"] = "TO" if all(r["timeout"] for r in runs) else f"F({successful}/3)"
            rows.append(row)
    expected = {(f, n, g, m) for f in FAMILIES for n in (512, 1024, 2048, 4096)
                for g in (0.99, 0.999) for m in METHODS}
    if len(rows) != 320 or {cell_key(r) for r in rows} != expected:
        raise ValueError("Expected all 320 distinct runtime cells")
    ranked = ranks(rows)
    return rows, ranked, {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def overall_positions(ranked):
    values = ranked["overall"]
    answer, i = {}, 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j]["mean_rank"] == values[i]["mean_rank"]:
            j += 1
        for row in values[i:j]:
            answer[row["method"]] = ((i + 1) + j) / 2
        i = j
    return answer


def table_data(rows, ranked, family=None):
    lookup = {cell_key(r): r for r in rows}
    data, bold = [], set()
    for f in ([family] if family else FAMILIES):
        for n in (512, 1024, 2048, 4096):
            for g in (0.99, 0.999):
                case = [lookup[(f, n, g, m)] for m in METHODS]
                minimum = min(r["median_wall_seconds"] for r in case if r["all_success"])
                index = len(data)
                data.append([f.title() if n == 512 and g == 0.99 else "",
                             str(n) if g == 0.99 else "", str(g)] + [r["display"] for r in case])
                bold.update((index, j + 3) for j, r in enumerate(case)
                            if r["all_success"] and r["median_wall_seconds"] == minimum)
    overall = {r["method"]: r for r in ranked["overall"]}
    if family:
        means = ranked["by_family"][family]
        success = {m: sum(r["family"] == family and r["method"] == m and r["all_success"]
                          for r in rows) for m in METHODS}
        summary = [("Family mean rank", [f'{means[m]:.4f}' for m in METHODS]),
                   ("Successful instances", [f'{success[m]}/8' for m in METHODS])]
    else:
        positions = overall_positions(ranked)
        summary = [("Mean rank (lower is better)", [f'{overall[m]["mean_rank"]:.4f}' for m in METHODS]),
                   ("Overall rank", [f'{positions[m]:g}' for m in METHODS]),
                   ("Successful instances", [f'{overall[m]["successful_configurations"]}/32' for m in METHODS])]
    return data, bold, summary


def notes(platform):
    common = ("Entries are median seconds from three runs. Bold marks the lowest successful time in each row, "
              "using unrounded values. Success requires all three runs to satisfy the independent residual "
              "check (infinity norm at most 1e-8). TO: all three runs timed out; F(s/3): only s runs succeeded. "
              "Per-run limits: 30 s for n at most 1024; 60 s otherwise. "
              "Ranks use unrounded times; unsuccessful methods tie below successful ones.")
    scope = ("These historical paper values combine multiple timing batches; per-cell provenance is retained. "
             if platform == "macos" else
             "This separate Linux run used the recorded NumPy Prescott / SciPy Haswell BLAS dispatch configuration. ")
    return common, scope + "Ranks are descriptive. These measurements are not an isolated CPU-performance comparison; see the archive metadata."


def draw_pdf(path, platform, rows, ranked, family=None):
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    # Matplotlib bundles DejaVu Serif. Embed it to keep the standalone PDF
    # portable, including on machines without a system font configuration.
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in (("TableRegular", "DejaVuSerif.ttf"),
                           ("TableBold", "DejaVuSerif-Bold.ttf"),
                           ("TableItalic", "DejaVuSerif-Italic.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    data, bold, summary = table_data(rows, ranked, family)
    width, height = 1224, (430 if family else 864)
    margin, table_width = 36, 1152
    col_widths = [68, 46, 50] + [(table_width - 164) / 10] * 10
    xpos = [margin]
    for w in col_widths:
        xpos.append(xpos[-1] + w)
    c = canvas.Canvas(str(path), pagesize=(width, height), invariant=1, pageCompression=1)
    c.setTitle("Runtime comparison - " + TITLES[platform] + (" - " + family.title() if family else ""))
    c.setAuthor("LSQN reproducibility package")
    c.setFont("TableBold", 20)
    c.drawString(margin, height - 40, "Runtime comparison | " + TITLES[platform])
    c.setFont("TableRegular", 11)
    c.drawString(margin, height - 59, SUBTITLES[platform] + (" | " + family.title() if family else ""))
    c.drawRightString(width - margin, height - 59, "Seconds | median of three runs")
    y = height - 77
    header_height, row_height = 27, 17
    c.setFillColor(colors.black)
    c.setLineWidth(0.9)
    c.line(margin, y, width - margin, y)
    headers = ["Problem", "n", "\u03b3"] + METHODS
    for j, text in enumerate(headers):
        c.setFont("TableItalic" if j in (1, 2) else ("TableBold" if j == 3 else "TableRegular"), 10)
        if j == 0:
            c.drawString(xpos[j] + 5, y - 18, text)
        else:
            c.drawRightString(xpos[j + 1] - 7, y - 18, text)
    y -= header_height
    c.setLineWidth(0.5)
    c.line(margin, y, width - margin, y)
    for i, row in enumerate(data):
        if i and i % 8 == 0:
            c.line(margin, y, width - margin, y)
        for j, text in enumerate(row):
            c.setFont("TableBold" if (i, j) in bold else "TableRegular", 10.5)
            if j == 0:
                c.drawString(xpos[j] + 5, y - 12, text)
            else:
                c.drawRightString(xpos[j + 1] - 7, y - 12, text)
        y -= row_height
    c.setLineWidth(0.7)
    c.line(margin, y, width - margin, y)
    for label, values in summary:
        c.setFont("TableRegular", 9.5)
        c.drawString(margin + 5, y - 14, label)
        for j, text in enumerate(values, 3):
            c.setFont("TableRegular", 10.5)
            c.drawRightString(xpos[j + 1] - 7, y - 14, text)
        y -= 20
    c.setLineWidth(1)
    c.line(margin, y, width - margin, y)
    style = ParagraphStyle("notes", fontName="TableRegular", fontSize=10, leading=13)
    y -= 13
    for text in notes(platform):
        p = Paragraph(text, style)
        _, h = p.wrap(table_width, height)
        y -= h
        if y < 18:
            raise ValueError("Runtime table notes overflow")
        p.drawOn(c, margin, y)
        y -= 6
    c.showPage()
    c.save()


def latex(platform, rows, ranked):
    data, bold, summary = table_data(rows, ranked)
    out = [r"% Requires \usepackage{booktabs,pdflscape}.",
           "% Generated from released benchmark records by scripts/plot_runtime_tables.py.",
           r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\begingroup",
           r"\fontsize{8}{9}\selectfont", r"\setlength{\tabcolsep}{3.8pt}",
           r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}lrr*{10}{r}@{}}",
           r"\toprule", r"Problem & $n$ & $\gamma$ & \textbf{LSQN} & " + " & ".join(METHODS[1:]) + r" \\",
           r"\midrule"]
    for i, row in enumerate(data):
        if i and i % 8 == 0:
            out.append(r"\midrule")
        out.append(" & ".join(r"\textbf{" + text + "}" if (i, j) in bold else text
                              for j, text in enumerate(row)) + r" \\")
    out.append(r"\midrule")
    for label, values in summary:
        label = label.replace(" (lower is better)", r" $\downarrow$")
        out.append(r"\multicolumn{3}{@{}l}{" + label + "} & " + " & ".join(values) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup",
                r"\caption{Runtime comparison on 32 discounted MDP instances (seconds): " + TITLES[platform] + ".}",
                r"\label{tab:runtime-" + platform + "}", r"\vspace{2pt}",
                r"\begin{minipage}{0.96\linewidth}", r"\fontsize{8}{9}\selectfont",
                r"\textit{Notes.} " + " ".join(notes(platform)).replace("1e-8", r"$10^{-8}$"),
                r"\end{minipage}", r"\end{table}", r"\end{landscape}"])
    return "\n".join(out) + "\n"


def png(pdf, path):
    # Capture output so a renderer failure remains readable; never ignore a
    # nonzero status or continue with an old PNG.
    result = subprocess.run(["pdftoppm", "-singlefile", "-r", "180", "-png", str(pdf), str(path.with_suffix(""))],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("pdftoppm failed: " + result.stderr[-4000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("all", "macos", "linux"), default="all")
    args = parser.parse_args()
    if not shutil.which("pdftoppm"):
        raise SystemExit("Install Poppler (pdftoppm) to render PNG files.")
    output = BENCH / "figures"
    output.mkdir(exist_ok=True)
    for platform in ("macos", "linux") if args.platform == "all" else (args.platform,):
        rows, ranked, sources = load_rows(platform)
        base = output / ("runtime_" + platform)
        draw_pdf(base.with_suffix(".pdf"), platform, rows, ranked)
        png(base.with_suffix(".pdf"), base.with_suffix(".png"))
        base.with_suffix(".tex").write_text(latex(platform, rows, ranked), encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="lsqn-runtime-") as tmp:
            for family in FAMILIES:
                pdf = Path(tmp) / (family + ".pdf")
                draw_pdf(pdf, platform, rows, ranked, family)
                png(pdf, output / (base.name + "_" + family + ".png"))
        metadata = dict(schema="lsqn-runtime-table-v1", platform=platform,
                        source_sha256=sources, configurations=32, cells=320,
                        ranks=ranked, failed_cells=[dict(family=r["family"], n=r["n"], gamma=r["gamma"],
                            method=r["method"], display=r["display"]) for r in rows if not r["all_success"]],
                        fastest_cells=[dict(family=r["family"], n=r["n"], gamma=r["gamma"], method=r["method"])
                            for r in rows if r["all_success"] and
                            r["median_wall_seconds"] == min(x["median_wall_seconds"] for x in rows
                            if x["all_success"] and cell_key(x)[:3] == cell_key(r)[:3])])
        base.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Rendered", platform, ": 320 cells,", len(metadata["fastest_cells"]), "row minima")


if __name__ == "__main__":
    main()
