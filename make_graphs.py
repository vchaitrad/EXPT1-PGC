import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq, omp, mpi, cuda_total, cuda_kernel = 645.066288, 415.637445, 222.147030, 0.363913, 0.294874
names = ["Sequential", "OpenMP\n(4 threads)", "MPI\n(4 processes)", "CUDA\n(T4 GPU)"]
times = [seq, omp, mpi, cuda_total]
speed = [seq/t for t in times]
colors = ["#6c757d", "#2a9d8f", "#e9a23b", "#e63946"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. Execution time
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, times, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Execution time (seconds, log scale)")
ax.set_title("Execution Time Comparison (4000 x 4000)", fontweight="bold")
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, t*1.15, f"{t:.3f} s", ha="center", fontweight="bold")
ax.set_ylim(0.1, 3000)
fig.tight_layout(); fig.savefig("graphs/execution_time.png", dpi=150); plt.close(fig)

# 2. Speedup
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, speed, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Speedup over sequential (log scale)")
ax.set_title("Speedup Comparison", fontweight="bold")
for b, s in zip(bars, speed):
    ax.text(b.get_x()+b.get_width()/2, s*1.15, f"{s:.2f}x", ha="center", fontweight="bold")
ax.set_ylim(0.5, 10000)
fig.tight_layout(); fig.savefig("graphs/speedup.png", dpi=150); plt.close(fig)

# 3. CPU parallel efficiency
fig, ax = plt.subplots(figsize=(6.5, 5))
eff = [speed[1]/4*100, speed[2]/4*100]
bars = ax.bar(["OpenMP\n(4 threads)", "MPI\n(4 processes)"], eff, color=["#2a9d8f", "#e9a23b"])
ax.axhline(100, color="gray", linestyle="--", linewidth=1)
ax.text(1.45, 102, "ideal = 100%", ha="right", color="gray", fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel("Parallel efficiency (%)")
ax.set_title("CPU Parallel Efficiency (speedup / 4 workers)", fontweight="bold")
for b, e in zip(bars, eff):
    ax.text(b.get_x()+b.get_width()/2, e+2, f"{e:.1f}%", ha="center", fontweight="bold")
fig.tight_layout(); fig.savefig("graphs/efficiency.png", dpi=150); plt.close(fig)

# 4. CUDA kernel vs total
fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(["Kernel only", "Total phase\n(with transfers)"], [cuda_kernel, cuda_total], color=["#e63946", "#f4a3a8"])
ax.set_ylabel("Time (seconds)"); ax.set_title("CUDA: Kernel vs Total Time", fontweight="bold")
for b, t in zip(bars, [cuda_kernel, cuda_total]):
    ax.text(b.get_x()+b.get_width()/2, t+0.008, f"{t:.6f} s", ha="center", fontweight="bold")
ax.set_ylim(0, 0.45)
fig.tight_layout(); fig.savefig("graphs/cuda_breakdown.png", dpi=150); plt.close(fig)

for n, s in zip(names, speed): print(n.replace("\n"," "), round(s,2))
print("cuda kernel speedup", round(seq/cuda_kernel,1))
print("eff", [round(e,1) for e in eff])
