"""
Experiment 2 - Multithreaded Programming with Pthreads and OpenMP
Performance-analysis figure generator.

Reads  : ../results/timing_results.csv
         ../results/sequential_runs.csv
         ../results/race_condition_results.csv
Writes : ../graphs/*.png

Usage  : python scripts/generate_graphs.py   (run from the Experiment_2 folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "graphs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data ----
df = pd.read_csv(os.path.join(RES, "timing_results.csv"))
seq = pd.read_csv(os.path.join(RES, "sequential_runs.csv"))
race = pd.read_csv(os.path.join(RES, "race_condition_results.csv"), keep_default_na=False)

T_SEQ = seq["Sequential_Time_s"].mean()
p = df["Threads"].to_numpy()
t_pt = df["Pthreads_Time_s"].to_numpy()
t_omp = df["OpenMP_Time_s"].to_numpy()
s_pt, s_omp = T_SEQ / t_pt, T_SEQ / t_omp
e_pt, e_omp = s_pt / p * 100, s_omp / p * 100

C_PT, C_OMP, C_SEQ, C_IDEAL = "#2563EB", "#16A34A", "#6B7280", "#111827"
C_BAD, C_OK = "#DC2626", "#16A34A"

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


# ------------------------------------------ 01 execution time vs threads --
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(p, t_pt, "o-", color=C_PT, lw=2.2, ms=8, label="Pthreads")
ax.plot(p, t_omp, "s-", color=C_OMP, lw=2.2, ms=8, label="OpenMP")
ax.axhline(T_SEQ, color=C_SEQ, ls="--", lw=1.5,
           label=f"Sequential baseline ({T_SEQ:.3f} s)")
for x, a, b in zip(p, t_pt, t_omp):
    ax.annotate(f"{a:.3f}", (x, a), xytext=(6, 8), textcoords="offset points",
                color=C_PT, fontsize=9.5)
    ax.annotate(f"{b:.3f}", (x, b), xytext=(-40, -16), textcoords="offset points",
                color=C_OMP, fontsize=9.5)
ax.set_xticks(p)
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 1.55)
ax.set_xlabel("Number of threads")
ax.set_ylabel("Execution time (seconds)")
ax.set_title("Fig. 1 - Execution Time vs Number of Threads")
ax.legend()
save(fig, "01_execution_time_vs_threads.png")

# ------------------------------------------------- 02 speedup vs threads --
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.plot([0, 16.5], [0, 16.5], ls=":", color=C_IDEAL, lw=1.3,
        label="Ideal speedup (= number of threads)")
ax.plot(p, s_pt, "o-", color=C_PT, lw=2.2, ms=8, label="Pthreads")
ax.plot(p, s_omp, "s-", color=C_OMP, lw=2.2, ms=8, label="OpenMP")
for x, a, b in zip(p, s_pt, s_omp):
    hi, lo = (a, b) if a >= b else (b, a)
    ca, cb = (C_PT, C_OMP) if a >= b else (C_OMP, C_PT)
    ax.annotate(f"{hi:.2f}x", (x, hi), xytext=(-12, 9), textcoords="offset points",
                color=ca, fontsize=9.5, fontweight="bold")
    ax.annotate(f"{lo:.2f}x", (x, lo), xytext=(8, -14), textcoords="offset points",
                color=cb, fontsize=9.5, fontweight="bold")
ax.set_xticks(p)
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 17.5)
ax.set_xlabel("Number of threads")
ax.set_ylabel("Speedup  (T_sequential / T_parallel)")
ax.set_title("Fig. 2 - Speedup vs Number of Threads")
ax.legend(loc="upper left")
save(fig, "02_speedup_vs_threads.png")

# ---------------------------------------------- 03 efficiency vs threads --
x = np.arange(len(p))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - w / 2, e_pt, w, color=C_PT, edgecolor="black", lw=0.6, label="Pthreads")
b2 = ax.bar(x + w / 2, e_omp, w, color=C_OMP, edgecolor="black", lw=0.6, label="OpenMP")
ax.axhline(100, color=C_IDEAL, ls=":", lw=1.3, label="Ideal (100 %)")
for b, v in list(zip(b1, e_pt)) + list(zip(b2, e_omp)):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center",
            fontsize=9.5, fontweight="bold")
ax.set_xticks(x, [f"{t} thread" + ("s" if t > 1 else "") for t in p])
ax.set_ylim(0, 125)
ax.set_ylabel("Efficiency  (Speedup / Threads)  %")
ax.set_title("Fig. 3 - Parallel Efficiency vs Number of Threads")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="x", visible=False)
save(fig, "03_efficiency_vs_threads.png")

# ------------------------------------- 04 Pthreads vs OpenMP head-to-head --
diff = (t_omp - t_pt) / t_pt * 100
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
b1 = a1.bar(x - w / 2, t_pt * 1000, w, color=C_PT, edgecolor="black", lw=0.6, label="Pthreads")
b2 = a1.bar(x + w / 2, t_omp * 1000, w, color=C_OMP, edgecolor="black", lw=0.6, label="OpenMP")
for b, v in list(zip(b1, t_pt)) + list(zip(b2, t_omp)):
    a1.text(b.get_x() + b.get_width() / 2, v * 1000 + 15, f"{v * 1000:.0f}",
            ha="center", fontsize=8.5)
a1.set_xticks(x, [str(t) for t in p])
a1.set_xlabel("Number of threads")
a1.set_ylabel("Execution time (milliseconds)")
a1.set_title("(a) Time side by side")
a1.legend()
a1.grid(axis="x", visible=False)
cols = [C_PT if d > 0 else C_OMP for d in diff]
a2.barh([str(t) for t in p], diff, color=cols, edgecolor="black", lw=0.6)
a2.axvline(0, color="black", lw=1)
for i, d in enumerate(diff):
    a2.text(d + (0.25 if d >= 0 else -0.25), i, f"{d:+.1f}%", va="center",
            ha="left" if d >= 0 else "right", fontweight="bold")
a2.set_xlim(-6.5, 6.5)
a2.set_xlabel("OpenMP time vs Pthreads time (%)")
a2.set_ylabel("Number of threads")
a2.set_title("(b) OpenMP time vs Pthreads time\n(+ : Pthreads faster,  - : OpenMP faster)", fontsize=12)
a2.grid(axis="y", visible=False)
fig.suptitle("Fig. 4 - Pthreads vs OpenMP: Same Work, Same Threads",
             fontweight="bold", fontsize=14)
fig.tight_layout()
save(fig, "04_pthreads_vs_openmp.png")

# --------------------------------------------- 05 race condition results --
labels = [f"{r.Program}\n({r.Synchronisation})" for r in race.itertuples()]
actual = race["Actual"].to_numpy()
expected = race["Expected"].to_numpy()
fig, ax = plt.subplots(figsize=(10.5, 5.5))
colors = [C_BAD if a != e else C_OK for a, e in zip(actual, expected)]
bars = ax.bar(labels, actual, color=colors, edgecolor="black", lw=0.6, width=0.55)
ax.axhline(expected[0], color=C_IDEAL, ls="--", lw=1.4,
           label=f"Expected value = {expected[0]:,}")
for b, a, e in zip(bars, actual, expected):
    lost = e - a
    tag = "correct" if lost == 0 else f"{lost:,} updates lost\n({lost / e * 100:.1f}%)"
    ax.text(b.get_x() + b.get_width() / 2, a + 8000, f"{a:,}\n{tag}",
            ha="center", fontsize=9.5, fontweight="bold")
ax.set_ylim(0, 520000)
ax.set_ylabel("Final counter value")
ax.set_title("Fig. 5 - Race Condition vs Synchronisation (4 threads x 100,000 increments)")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
save(fig, "05_race_condition_results.png")

# ----------------------------------------- 06 ideal vs measured (overhead) --
ideal = T_SEQ / p
fig, ax = plt.subplots(figsize=(10, 5.5))
w3 = 0.27
ax.bar(x - w3, ideal * 1000, w3, color="#D1D5DB", edgecolor="black", lw=0.6,
       label="Ideal time (T_seq / threads)")
ax.bar(x, t_pt * 1000, w3, color=C_PT, edgecolor="black", lw=0.6, label="Pthreads (measured)")
ax.bar(x + w3, t_omp * 1000, w3, color=C_OMP, edgecolor="black", lw=0.6, label="OpenMP (measured)")
ax.set_yscale("log")
ax.set_xticks(x, [f"{t}" for t in p])
ax.set_xlabel("Number of threads")
ax.set_ylabel("Execution time (ms, log scale)")
ax.set_title("Fig. 6 - Ideal vs Measured Time (the gap is parallel overhead)")
for i in range(len(p)):
    gap = (t_omp[i] - ideal[i]) * 1000
    ax.text(x[i], max(t_pt[i], t_omp[i]) * 1000 * 1.18, f"OpenMP gap\n{gap:+.0f} ms",
            ha="center", fontsize=8.5, color="#374151")
ax.set_ylim(50, 3000)
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "06_ideal_vs_measured_time.png")

# --------------------------------------------------- 07 summary dashboard --
fig = plt.figure(figsize=(14, 7.8))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.3], hspace=0.4, wspace=0.3)
kpis = [("Best time (16 threads)", f"{t_pt[-1] * 1000:.1f} ms", "Pthreads", C_PT),
        ("Best speedup", f"{max(s_pt[-1], s_omp[-1]):.2f}x", "Pthreads, 16 threads", C_PT),
        ("Race fixed by", "mutex / critical", "400,000 = 400,000", C_OK)]
for i, (title, big, small, c) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#F9FAFB", edgecolor=c, lw=2.5))
    ax.text(0.5, 0.76, title.upper(), ha="center", fontsize=11, color="#374151",
            transform=ax.transAxes)
    ax.text(0.5, 0.42, big, ha="center", fontsize=22, fontweight="bold",
            color=c, transform=ax.transAxes)
    ax.text(0.5, 0.13, small, ha="center", fontsize=11, color="#374151",
            transform=ax.transAxes)
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
rows = [[str(t), f"{a:.6f}", f"{b:.6f}", f"{sa:.2f}x", f"{sb:.2f}x", f"{ea:.1f}%", f"{eb:.1f}%"]
        for t, a, b, sa, sb, ea, eb in zip(p, t_pt, t_omp, s_pt, s_omp, e_pt, e_omp)]
tbl = ax.table(cellText=rows,
               colLabels=["Threads", "Pthreads\ntime (s)", "OpenMP\ntime (s)",
                          "Pthreads\nspeedup", "OpenMP\nspeedup",
                          "Pthreads\nefficiency", "OpenMP\nefficiency"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(cell.get_height() * 1.6)
    elif c in (1, 3, 5):
        cell.set_facecolor("#EFF6FF")
    elif c in (2, 4, 6):
        cell.set_facecolor("#F0FDF4")
ax.set_title(f"Sequential baseline = {T_SEQ:.6f} s  (average of {len(seq)} runs)",
             fontweight="bold")
fig.suptitle("Fig. 7 - Experiment 2 Performance Summary  (sum of 10^9 terms)",
             fontsize=15, fontweight="bold")
save(fig, "07_performance_summary.png")

print("\nSequential baseline:", round(T_SEQ, 6))
for t, a, b, c, d in zip(p, s_pt, s_omp, e_pt, e_omp):
    print(f"{t:>2} thr | speedup P {a:.3f}  O {b:.3f} | eff P {c:.2f}%  O {d:.2f}%")
print("All figures written to", OUT)
