"""Replot archived trajectories in the paper's four-panel format.

Run after installing the project dependencies. Each curve uses the repetition
with median wall time; no trajectory is interpolated, smoothed, or extrapolated.
The macOS curves use the final runtime table's source batches, not the older
figure-profile batches. Both hosts therefore follow the default paper profile.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
METHODS = ["LSQN", "VI", "PI", "MPI-4", "MPI-20", "QPI-U", "QPI-BT", "QPI-S", "iGMRES-PI", "AA-I-S-5"]
FAMILIES = ["queue", "cluster", "garnet", "latent"]
SIZES = [512, 1024, 2048, 4096]
COLORS = ["#c0392b", "#7f8c8d", "#154360", "#00897b", "#8e44ad", "#b9770e", "#e67e22", "#a64d79", "#2471a3", "#4d7731"]


def read(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def load_linux(directory):
    if read(directory / "protocol.json").get("requested_options", {}).get("implementation") != "paper":
        raise ValueError("These paired figures require the default paper profile")
    groups = defaultdict(list)
    for row in read(directory / "summary.json")["runs"]:
        groups[row["case_id"], row["method"]].append(row)
    curves = []
    for (case, method), rows in sorted(groups.items()):
        if len(rows) != 3 or {r["repeat"] for r in rows} != {0, 1, 2}:
            raise ValueError("Incomplete timing triple: " + case)
        selected = sorted(rows, key=lambda r: (r["wall_seconds"], r["repeat"]))[1]
        raw = read(directory / selected["file"])
        curves.append(dict(family=raw["spec"]["family"], n=raw["spec"]["n"],
            gamma=raw["spec"]["gamma"], method=method, selected_repeat=selected["repeat"],
            all_success=all(r["success"] for r in rows), wall_seconds=selected["wall_seconds"],
            trace=[dict(k=t["k"], residual2=t["residual2"]) for t in raw["trace"]],
            source_ref=dict(published_path="benchmarks/linux-intel-2026-10-25/main/" + selected["file"] + ".gz")))
    return curves


def load_mac():
    with gzip.open(BENCH / "macos-m2-paper/main_curves.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(rows):
    keys = [(r["family"], r["n"], r["gamma"], r["method"]) for r in rows]
    expected = {(f, n, g, m) for f in FAMILIES for n in SIZES for g in (0.99, 0.999) for m in METHODS}
    if len(keys) != 320 or set(keys) != expected:
        raise ValueError("Expected exactly 320 distinct method/configuration curves")
    for row in rows:
        ks = [t["k"] for t in row["trace"]]
        if not ks or ks != sorted(ks):
            raise ValueError("Missing or unordered trace")


def values(row):
    x = np.asarray([t["k"] for t in row["trace"]], dtype=float)
    y = np.asarray([np.nan if t["residual2"] is None else t["residual2"] for t in row["trace"]], dtype=float)
    return x, np.where(np.isfinite(y), np.maximum(y, 1e-16), np.nan)


def plot(data, output):
    plt.rcParams.update({"font.family":"DejaVu Serif", "font.size":11, "axes.labelsize":12,
        "axes.titlesize":13, "pdf.fonttype":42, "ps.fonttype":42, "axes.spines.top":False,
        "axes.spines.right":False, "axes.linewidth":0.7, "path.simplify":False})
    output.mkdir(parents=True, exist_ok=True)
    indexed = {host: {(r["family"], r["n"], r["gamma"], r["method"]):r for r in rows}
               for host, rows in data.items()}
    outputs = []
    for family in FAMILIES:
        limits = {}
        for n in SIZES:
            series = [values(row) for rows in data.values() for row in rows if row["family"] == family and row["n"] == n]
            series = [(x,y) for x,y in series if np.any(np.isfinite(y))]
            maxk = max(float(x.max()) for x,y in series)
            minr = min(1e-8, min(float(np.nanmin(y)) for x,y in series))
            maxr = max(float(np.nanmax(y)) for x,y in series)
            limits[n] = (max(10, maxk*1.12), max(1e-16, 10**np.floor(np.log10(minr))*0.7), maxr*2)
        for host in data:
            fig, axes = plt.subplots(2,2,figsize=(9.2,7.8))
            for ax,n in zip(axes.flat,SIZES):
                for method,color in zip(METHODS,COLORS):
                    for gamma,style in [(0.99,"-"),(0.999,"--")]:
                        row=indexed[host][family,n,gamma,method]
                        x,y=values(row)
                        finite_indices = np.flatnonzero(np.isfinite(y))
                        if not len(finite_indices):
                            continue
                        ax.plot(x,y,style,color=color,lw=1.65 if method=="LSQN" else 1.1,
                            alpha=1 if method=="LSQN" else .9,zorder=5 if method=="LSQN" else 2)
                        last = finite_indices[-1]
                        ax.plot(x[last],y[last],marker="o" if row["all_success"] else "x",color=color,
                            ms=3.5 if row["all_success"] else 6,mew=.8 if row["all_success"] else 1.4,linestyle="none")
                xmax,ymin,ymax=limits[n]
                ax.set_yscale("log")
                ax.set_xscale("symlog",linthresh=1,linscale=.8)
                ax.set_xticks([t for t in [0,1,10,100,1000,10000] if t <= xmax])
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xlim(0,xmax)
                ax.set_ylim(ymin,ymax)
                ax.set_title(f"$n={n}$",pad=9)
                ax.set_xlabel("Outer iteration $k$ (symlog scale)")
                ax.set_ylabel(r"$\|F(v_k)\|_2$")
                ax.grid(True,which="major",alpha=.17,lw=.6)
            handles=[Line2D([0],[0],color=c,lw=2,label=m) for m,c in zip(METHODS,COLORS)]
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.985),ncol=5,
                frameon=False,fontsize=10,columnspacing=1.4,handlelength=2)
            fig.text(.5,.887,r"Solid: $\gamma=0.99$     Dashed: $\gamma=0.999$     $\times$: failed triple / time limit",ha="center",fontsize=10)
            fig.subplots_adjust(top=.84,bottom=.08,left=.095,right=.985,hspace=.42,wspace=.27)
            stem=f"{family}_{host}"
            for extension in ("pdf","png"):
                path=output/(stem+"."+extension)
                fig.savefig(path,dpi=300,bbox_inches="tight",pad_inches=.08,
                    metadata={"Creator":"LSQN benchmark plotter", "CreationDate":None} if extension=="pdf" else {})
                outputs.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            plt.close(fig)
    provenance=dict(schema="lsqn-cross-platform-figures-v1",curve_count_per_host=320,
        profile="Final runtime-table configurations; macOS historical batches and Linux default paper profile.",
        style="Paper colors, 2x2 state-size panels, symlog x, log l2 residual y; equal limits for each paired panel; no main title or bottom footnote.",
        selection="Median wall-time repetition, ordered by (wall_seconds, repeat); unsuccessful triples retained and marked x.",
        residual_display_floor=1e-16,nonfinite_display="Nonfinite residuals appear as gaps; finite values below the display floor are clamped. Complete records remain archived.",
        png_dpi=300,pdf="Vector curves and embedded fonts",outputs=outputs,
        curves={host:[{k:v for k,v in r.items() if k!="trace"} | {"points":len(r["trace"])} for r in rows] for host,rows in data.items()})
    (output/"curves_provenance.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data={"macos":load_mac(),"linux":load_linux(BENCH/"linux-intel-2026-10-25/main")}
    for rows in data.values():
        validate(rows)
    plot(data,BENCH/"figures")
    print("Created eight paired residual figures (PDF and 300 dpi PNG).")


if __name__=="__main__":
    main()
