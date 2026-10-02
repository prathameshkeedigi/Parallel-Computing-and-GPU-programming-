"""
Experiment 1 - Parallel Matrix Multiplication (4000 x 4000)
Performance-analysis figure generator.

Reads  : ../results/timing_results.csv
Writes : ../graphs/*.png

Usage  : python scripts/generate_graphs.py   (run from the Experiment_1 folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "results", "timing_results.csv")
OUT = os.path.join(ROOT, "graphs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data ----
df = pd.read_csv(CSV)
N = 4000
FLOP = 2 * N ** 3                      # one multiply + one add per inner iteration
T = dict(zip(df["Implementation"], df["Time_s"]))
T_SEQ = T["Sequential"]
T_KERNEL = float(df.loc[df["Implementation"] == "CUDA", "Kernel_Time_s"].iloc[0])
T_XFER = T["CUDA"] - T_KERNEL

# Reference values printed in the lab manual (used only for comparison).
REF = {"Sequential": 244.120000, "OpenMP": 30.830434,
       "MPI": 92.979510, "CUDA": 0.165004}

NAMES = ["Sequential", "OpenMP", "MPI", "CUDA"]
LABELS = ["Sequential\n(1 core)", "OpenMP\n(8 threads)",
          "MPI\n(4 VMs)", "CUDA\n(GPU)"]
COL = {"Sequential": "#6B7280", "OpenMP": "#2563EB",
       "MPI": "#EA580C", "CUDA": "#16A34A"}
COLORS = [COL[n] for n in NAMES]
times = np.array([T[n] for n in NAMES])
speedup = T_SEQ / times
gflops = FLOP / times / 1e9

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.linestyle": "--", "grid.alpha": 0.35,
    "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(fig)
    print("saved", name)


def fmt_time(t):
    return f"{t:.3f} s" if t < 1 else f"{t:.2f} s"


# ------------------------------------------- 01 execution time (log) ----
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(LABELS, times, color=COLORS, width=0.6, edgecolor="black", lw=0.6)
ax.set_yscale("log")
ax.set_ylim(0.03, 1500)
ax.set_ylabel("Execution time (seconds, log scale)")
ax.set_title("Fig. 1 - Execution Time of 4000 x 4000 Matrix Multiplication")
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, t * 1.25, fmt_time(t),
            ha="center", va="bottom", fontweight="bold")
ax.grid(axis="x", visible=False)
save(fig, "01_execution_time_comparison.png")

# ------------------------------------------------- 02 speedup (log) -----
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(LABELS, speedup, color=COLORS, width=0.6, edgecolor="black", lw=0.6)
ax.set_yscale("log")
ax.set_ylim(0.5, 20000)
ax.axhline(1, color="black", lw=1, ls=":")
ax.set_ylabel("Speedup over sequential (x, log scale)")
ax.set_title("Fig. 2 - Speedup Relative to the Sequential Baseline")
for b, s in zip(bars, speedup):
    ax.text(b.get_x() + b.get_width() / 2, s * 1.25, f"{s:,.2f}x",
            ha="center", va="bottom", fontweight="bold")
ax.grid(axis="x", visible=False)
save(fig, "02_speedup_comparison.png")

# ------------------------------------------------ 03 throughput ---------
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(LABELS[::-1], gflops[::-1], color=COLORS[::-1],
               edgecolor="black", lw=0.6, height=0.6)
ax.set_xscale("log")
ax.set_xlim(0.1, 20000)
ax.set_xlabel("Sustained throughput (GFLOP/s, log scale)")
ax.set_title(f"Fig. 3 - Computational Throughput  (workload = 2N³ = {FLOP:.2e} FLOP)")
for b, g in zip(bars, gflops[::-1]):
    ax.text(g * 1.2, b.get_y() + b.get_height() / 2, f"{g:,.2f} GFLOP/s",
            va="center", fontweight="bold")
ax.grid(axis="y", visible=False)
save(fig, "03_throughput_gflops.png")

# ------------------------------------- 04 parallel efficiency (CPU) -----
p_omp, p_mpi = 8, 4
S_omp, S_mpi = speedup[1], speedup[2]
E_omp, E_mpi = S_omp / p_omp, S_mpi / p_mpi
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(2)
a1.bar(x - 0.18, [p_omp, p_mpi], 0.36, color="#D1D5DB", edgecolor="black",
       lw=0.6, label="Ideal (linear) speedup")
a1.bar(x + 0.18, [S_omp, S_mpi], 0.36, color=[COL["OpenMP"], COL["MPI"]],
       edgecolor="black", lw=0.6, label="Measured speedup")
for xi, s in zip(x, [S_omp, S_mpi]):
    a1.text(xi + 0.18, s + 0.15, f"{s:.2f}x", ha="center", fontweight="bold")
for xi, p in zip(x, [p_omp, p_mpi]):
    a1.text(xi - 0.18, p + 0.15, f"{p}x", ha="center")
a1.set_xticks(x, ["OpenMP (p = 8)", "MPI (p = 4)"])
a1.set_ylabel("Speedup (x)")
a1.set_title("(a) Ideal vs measured speedup")
a1.legend(loc="upper right")
a1.grid(axis="x", visible=False)
a2.bar(["OpenMP (p = 8)", "MPI (p = 4)"], [E_omp * 100, E_mpi * 100],
       color=[COL["OpenMP"], COL["MPI"]], edgecolor="black", lw=0.6, width=0.5)
a2.axhline(100, color="black", ls=":", lw=1)
a2.set_ylim(0, 110)
a2.set_ylabel("Parallel efficiency  E = S / p  (%)")
a2.set_title("(b) Parallel efficiency")
for i, e in enumerate([E_omp, E_mpi]):
    a2.text(i, e * 100 + 2, f"{e * 100:.1f} %", ha="center", fontweight="bold")
a2.grid(axis="x", visible=False)
fig.suptitle("Fig. 4 - Scalability of the CPU Parallel Models", fontweight="bold", fontsize=14)
fig.tight_layout()
save(fig, "04_parallel_efficiency.png")

# ------------------------------ 05 Amdahl + Karp-Flatt serial fraction --
def karp_flatt(S, p):
    return (1 / S - 1 / p) / (1 - 1 / p)


f_omp, f_mpi = karp_flatt(S_omp, p_omp), karp_flatt(S_mpi, p_mpi)
p = np.arange(1, 33)
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.plot(p, p, color="black", ls=":", lw=1.2, label="Ideal linear speedup")
for f, c, n in [(f_omp, COL["OpenMP"], "OpenMP"), (f_mpi, COL["MPI"], "MPI")]:
    ax.plot(p, 1 / (f + (1 - f) / p), color=c, lw=2,
            label=f"Amdahl fit for {n}  (effective serial fraction e = {f:.3f})")
ax.scatter([p_omp], [S_omp], s=120, color=COL["OpenMP"], edgecolor="black", zorder=5)
ax.scatter([p_mpi], [S_mpi], s=120, color=COL["MPI"], edgecolor="black", zorder=5)
ax.annotate(f"Measured OpenMP\n{S_omp:.2f}x @ 8 threads", (p_omp, S_omp),
            xytext=(12, 4.0), arrowprops=dict(arrowstyle="->"))
ax.annotate(f"Measured MPI\n{S_mpi:.2f}x @ 4 processes", (p_mpi, S_mpi),
            xytext=(8, 1.6), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(1, 32)
ax.set_ylim(0, 33)
ax.set_xlabel("Number of processing elements p (threads / processes)")
ax.set_ylabel("Speedup S(p)")
ax.set_title("Fig. 5 - Amdahl's Law Projection using Karp-Flatt Serial Fraction")
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "05_amdahl_karp_flatt.png")

# --------------------------------------- 06 CUDA time breakdown ---------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
parts = [T_KERNEL * 1000, T_XFER * 1000]
a1.pie(parts, labels=["Kernel execution", "PCIe transfers\n(H2D A,B + D2H C)"],
       colors=[COL["CUDA"], "#A7F3D0"], autopct="%1.1f %%", startangle=90,
       wedgeprops=dict(width=0.42, edgecolor="white"), pctdistance=0.78,
       textprops=dict(fontsize=11))
a1.text(0, 0, f"{T['CUDA'] * 1000:.2f} ms\ntotal", ha="center", va="center",
        fontsize=12, fontweight="bold")
a1.set_title("(a) Composition of total CUDA phase time")
a2.barh(["CUDA phase"], [parts[0]], color=COL["CUDA"], edgecolor="black",
        lw=0.6, height=0.45, label=f"Kernel  {parts[0]:.3f} ms")
a2.barh(["CUDA phase"], [parts[1]], left=[parts[0]], color="#A7F3D0",
        edgecolor="black", lw=0.6, height=0.45, label=f"Transfers  {parts[1]:.3f} ms")
a2.set_xlabel("Time (ms)")
a2.set_title("(b) Timeline view")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
a2.grid(axis="y", visible=False)
bw = 3 * N * N * 4 / T_XFER / 1e9
a2.set_ylim(-0.6, 0.75)
a2.text(sum(parts) / 2, 0.42,
        f"192 MB moved over PCIe  ->  ~{bw:.1f} GB/s effective", ha="center")
fig.suptitle("Fig. 6 - CUDA Kernel vs Data-Transfer Time", fontweight="bold", fontsize=14)
fig.tight_layout()
save(fig, "06_cuda_time_breakdown.png")

# --------------------------------------- 07 MPI overhead analysis -------
ideal_mpi = T_SEQ / p_mpi
overhead = T["MPI"] - ideal_mpi
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.barh(["Measured MPI", "Ideal MPI (T_seq / 4)"], [T["MPI"], ideal_mpi],
        color=[COL["MPI"], "#FED7AA"], edgecolor="black", lw=0.6, height=0.5)
ax.barh(["Measured MPI"], [overhead], left=[ideal_mpi], color="none",
        edgecolor="black", hatch="///", lw=0.6, height=0.5)
ax.axvline(T_SEQ, color=COL["Sequential"], ls="--", lw=1.5)
ax.text(T_SEQ - 4, 1.45, f"Sequential\n{T_SEQ:.1f} s", ha="right",
        color=COL["Sequential"], fontweight="bold")
ax.text(ideal_mpi / 2, 0, f"ideal share\n{ideal_mpi:.1f} s", ha="center",
        va="center", color="white", fontweight="bold")
ax.text(ideal_mpi + overhead / 2, 0,
        f"overhead\n{overhead:.1f} s ({overhead / T['MPI'] * 100:.0f} %)",
        ha="center", va="center", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
ax.text(ideal_mpi + 3, 1, f"{ideal_mpi:.1f} s", va="center")
ax.set_xlim(0, 300)
ax.set_ylim(-0.5, 1.9)
ax.set_xlabel("Time (seconds)")
ax.set_title("Fig. 7 - MPI: Ideal Compute Share vs Observed Overhead")
ax.grid(axis="y", visible=False)
save(fig, "07_mpi_overhead_analysis.png")

# -------------------------------- 08 measured vs reference manual -------
ref = np.array([REF[n] for n in NAMES])
x = np.arange(4)
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - 0.2, ref, 0.4, color="#E5E7EB", edgecolor="black", lw=0.6,
            label="Lab-manual reference")
b2 = ax.bar(x + 0.2, times, 0.4, color=COLORS, edgecolor="black", lw=0.6,
            label="This experiment (measured)")
ax.set_yscale("log")
ax.set_ylim(0.03, 2000)
ax.set_xticks(x, LABELS)
ax.set_ylabel("Execution time (seconds, log scale)")
ax.set_title("Fig. 8 - Measured Results vs Lab-Manual Reference")
for b, t in list(zip(b1, ref)) + list(zip(b2, times)):
    ax.text(b.get_x() + b.get_width() / 2, t * 1.2, fmt_time(t), ha="center",
            va="bottom", fontsize=8.5, rotation=0)
ax.legend(handles=[Patch(facecolor="#E5E7EB", edgecolor="black", label="Lab-manual reference"),
                   Patch(facecolor="#9CA3AF", edgecolor="black", label="This experiment (measured)")],
          loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "08_measured_vs_reference.png")

# --------------------------------------------- 09 summary dashboard -----
fig = plt.figure(figsize=(14, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45, wspace=0.32)
kpis = [("Fastest model", "CUDA", f"{T['CUDA'] * 1000:.1f} ms end-to-end", COL["CUDA"]),
        ("Peak speedup", f"{speedup[3]:,.0f}x", f"kernel-only {T_SEQ / T_KERNEL:,.0f}x", COL["CUDA"]),
        ("Best CPU model", f"OpenMP {S_omp:.2f}x", f"efficiency {E_omp * 100:.1f} %", COL["OpenMP"])]
for i, (title, big, small, c) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#F9FAFB", edgecolor=c, lw=2.5))
    ax.text(0.5, 0.78, title.upper(), ha="center", fontsize=11, color="#374151",
            transform=ax.transAxes)
    ax.text(0.5, 0.45, big, ha="center", fontsize=24, fontweight="bold",
            color=c, transform=ax.transAxes)
    ax.text(0.5, 0.17, small, ha="center", fontsize=11, color="#374151",
            transform=ax.transAxes)
ax = fig.add_subplot(gs[1, :2])
ax.axis("off")
rows = [[n, fmt_time(t), f"{s:,.2f}x", f"{g:,.2f}", "4000.00"]
        for n, t, s, g in zip(NAMES, times, speedup, gflops)]
tbl = ax.table(cellText=rows, colLabels=["Model", "Time", "Speedup", "GFLOP/s", "C[0][0]"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1, 2.1)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor(COL[NAMES[r - 1]])
        cell.set_text_props(color="white", fontweight="bold")
ax.set_title("Result summary (all outputs verified)", fontweight="bold")
ax = fig.add_subplot(gs[1, 2])
ax.bar(["Seq", "OMP", "MPI", "CUDA"], speedup, color=COLORS, edgecolor="black", lw=0.6)
ax.set_yscale("log")
ax.set_title("Speedup (log)", fontweight="bold")
ax.grid(axis="x", visible=False)
fig.suptitle("Fig. 9 - Experiment 1 Performance Dashboard  (4000 x 4000, FP64 CPU / FP32 GPU)",
             fontsize=15, fontweight="bold")
save(fig, "09_performance_dashboard.png")

print("\nAll figures written to", OUT)
