"""Create the paired coverage figure from the archived IRIDIA ArNet2 results.

python make_figure1.py --results /path/to/holter_intervals/results --out .
"""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

FIXED = "fixed_radius"
ADAPTIVE = "scaled_radius"
ORANGE = "#AB451E"
BLUE = "#225577"
INK = "#172C3B"
MUTED = "#4D6270"
HASHES = {
    "patients.csv": "1e6cda51bd80e92a7e97556541df745853a8991cd5056e9d51613704a205a135",
    "records_IRIDIA_ArNet2.csv": "723480d42323b00b12d42775624bdf0f3a77853045166b17db5430e11242b077",
}

def verified_rows(results, name):
    path = results / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f"{name} differs from the deposited study")
    with path.open(newline="") as source:
        return list(csv.DictReader(source))

def patient_pairs(results):
    records = defaultdict(list)
    for row in verified_rows(results, "records_IRIDIA_ArNet2.csv"):
        if row["method"] in (FIXED, ADAPTIVE):
            records[(row["patient"], row["method"])].append(float(row["coverage"]))

    patients = defaultdict(dict)
    for row in verified_rows(results, "patients.csv"):
        if (row["cohort"], row["detector"]) != ("IRIDIA", "ArNet2"):
            continue
        patient, method = row["patient"], row["method"]
        if method not in (FIXED, ADAPTIVE):
            continue
        value = float(row["coverage"])
        assert method not in patients[patient]
        assert abs(value - mean(records[(patient, method)])) < 1e-12
        patients[patient][method] = value

    result = []
    for patient, values in sorted(patients.items()):
        number = len(records[(patient, FIXED)])
        assert number == len(records[(patient, ADAPTIVE)])
        result.append((patient, number, values[FIXED]*100, values[ADAPTIVE]*100))
    assert len(result) == 152 and sum(row[1] for row in result) == 167
    return result

def quadrant(fixed, adaptive):
    if fixed >= 80 and adaptive >= 80:
        return "both_at_or_above_80"
    if fixed >= 80:
        return "fixed_only_at_or_above_80"
    if adaptive >= 80:
        return "adaptive_only_at_or_above_80"
    return "both_below_80"

def export_pairs(pairs, directory):
    with (directory / "Figure1_patient_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "records", "fixed_coverage_pct",
                         "adaptive_coverage_pct", "threshold_category"])
        for patient, n, x, y in pairs:
            writer.writerow([patient, n, f"{x:.6f}", f"{y:.6f}", quadrant(x,y)])

def figure(pairs, directory):
    counts = Counter(quadrant(x, y) for _, _, x, y in pairs)
    perfect = sum(x == 100 and y == 100 for _, _, x, y in pairs)
    assert tuple(counts[k] for k in ("both_at_or_above_80", "both_below_80",
                 "fixed_only_at_or_above_80", "adaptive_only_at_or_above_80")) == (138, 10, 4, 0)
    assert perfect == 82
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.labelcolor": INK,
                         "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(7.04, 3.83))
    ax = fig.add_axes([0.104, 0.160, 0.400, 0.735])
    side = fig.add_axes([0.620, 0.240, 0.358, 0.610])
    for active in (ax, side):
        active.spines[["top", "right"]].set_visible(False)
        for key in ("left", "bottom"):
            active.spines[key].set_color("#728593")
            active.spines[key].set_linewidth(0.6)
        active.tick_params(length=3, width=.6, color="#728593", labelcolor=MUTED)
        active.set_axisbelow(True)
    fig.text(.024,.945,"A",fontsize=12,weight="bold")
    fig.text(.071,.945,"All 152 patients",fontsize=10.5,weight="bold")
    fig.text(.550,.945,"B",fontsize=12,weight="bold")
    fig.text(.594,.945,"Four downward crossings",fontsize=10.5,weight="bold")
    ax.set_aspect("equal")
    ax.set(xlim=(20,104),ylim=(20,104),xlabel="Coverage with fixed radius (%)",
           ylabel="Coverage with adaptive radius (%)")
    ticks = [20,40,60,80,100]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.plot([20,104],[20,104],color="#BCC6CB",lw=.7,zorder=0,clip_on=True)
    ax.axhline(80,color="#7F8F99",ls=(0,(3,2.3)),lw=.72,zorder=1)
    ax.axvline(80,color="#7F8F99",ls=(0,(3,2.3)),lw=.72,zorder=1)
    # Count all patients, then draw each coincident location at its exact position.
    locations = Counter((x,y) for _,_,x,y in pairs)
    for x, y in locations:
        crosses = x >= 80 and y < 80
        if x == y == 100:
            ax.plot(x,y,marker="o",ms=5.8,mfc="white",mec=BLUE,mew=1.35,zorder=3)
        else:
            ax.scatter(x,y,s=27 if crosses else 15,marker="D" if crosses else "o",
                       facecolor=ORANGE if crosses else BLUE,edgecolor="white",
                       lw=.36,zorder=4 if crosses else 2,alpha=.96 if crosses else .76)
    ax.annotate("82 at (100, 100)",xy=(100,100),xytext=(60,100.6),
                fontsize=7.1,ha="center",va="center",color=BLUE,
                arrowprops={"arrowstyle":"-", "lw":.65, "color":BLUE,
                            "shrinkA":1.5,"shrinkB":3},zorder=5)
    for x, y, count in [(23,84,0),(83,84,138),(23,26,10),(83,26,4)]:
        ax.text(x,y,f"n = {count}",fontsize=7.1,color=MUTED,va="center",
                bbox={"facecolor":"white","pad":.3,"edgecolor":"none","alpha":.9})
    crossed = sorted((r for r in pairs if r[2]>=80 and r[3]<80),key=lambda r:r[2])
    side.spines["left"].set_visible(False)
    side.tick_params(axis="y",length=0,pad=3)
    side.set_xlim(20,104)
    side.set_ylim(-.65,3.85)
    side.xaxis.set_major_locator(FixedLocator(ticks))
    side.set_xlabel("Patient coverage (%)")
    side.axvline(80,color="#7F8F99",ls=(0,(3,2.3)),lw=.72)
    ypos = list(range(3,-1,-1))
    side.set_yticks(ypos,[row[0].split("_")[-1] for row in crossed])
    for (_,_,fixed,adaptive), y in zip(crossed,ypos):
        side.plot([adaptive,fixed],[y,y],color="#AAB8C0",lw=1.5,zorder=1)
        side.plot(fixed,y,"o",ms=5,color=BLUE,zorder=2)
        side.plot(adaptive,y,"D",ms=5,color=ORANGE,zorder=2)
        for value,color in [(fixed,BLUE),(adaptive,ORANGE)]:
            side.text(value,y+.17,f"{value:.1f}",fontsize=7.6,color=color,
                      va="bottom",ha="center")
    fig.text(.596,.867,"ID",size=7.3,color=MUTED)
    legend_y = .860
    for xpos,label,marker,color in [(.685,"Adaptive","D",ORANGE),(.846,"Fixed","o",BLUE)]:
        side.plot([xpos],[legend_y],marker=marker,markersize=4.9,color=color,
                  transform=fig.transFigure,clip_on=False)
        fig.text(xpos+.013,legend_y,label,fontsize=8,va="center",color=color)
    pdf = directory / "Figure1_paired_patient_coverage.pdf"
    fig.savefig(pdf,metadata={"Title":"Patient coverage of AF burden intervals with fixed and adaptive radii"})
    fig.savefig(directory / "Figure1_paired_patient_coverage.png",dpi=450)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("."))
    options = parser.parse_args()
    options.out.mkdir(parents=True,exist_ok=True)
    rows = patient_pairs(options.results)
    export_pairs(rows, options.out)
    figure(rows, options.out)
