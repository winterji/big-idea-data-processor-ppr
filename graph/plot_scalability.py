import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ================= DATA =================
# Doplněna data pro "CPU Parallel (No SIMD)" z tvých logů
data = [
    # --- DEXCOM (~37k prvků) ---
    {"Dataset": "Dexcom", "Size": 3.6e4, "Algorithm": "CPU Sequential", "Time": 5.092},
    {"Dataset": "Dexcom", "Size": 3.6e4, "Algorithm": "CPU Parallel (No SIMD)", "Time": 4.606},
    {"Dataset": "Dexcom", "Size": 3.6e4, "Algorithm": "CPU Parallel + SIMD", "Time": 0.875},
    {"Dataset": "Dexcom", "Size": 3.6e4, "Algorithm": "GPU (OpenCL)", "Time": 127.735},

    # --- HR (~11M prvků) ---
    {"Dataset": "HR", "Size": 1.1e7, "Algorithm": "CPU Sequential", "Time": 1573.03},
    {"Dataset": "HR", "Size": 1.1e7, "Algorithm": "CPU Parallel (No SIMD)", "Time": 1396.68},
    {"Dataset": "HR", "Size": 1.1e7, "Algorithm": "CPU Parallel + SIMD", "Time": 230.387},
    {"Dataset": "HR", "Size": 1.1e7, "Algorithm": "GPU (OpenCL)", "Time": 19.66},

    # --- BVP (~740M prvků) ---
    {"Dataset": "BVP", "Size": 7.3e8, "Algorithm": "CPU Sequential", "Time": 81742.2},
    {"Dataset": "BVP", "Size": 7.3e8, "Algorithm": "CPU Parallel (No SIMD)", "Time": 68383.3},
    {"Dataset": "BVP", "Size": 7.3e8, "Algorithm": "CPU Parallel + SIMD", "Time": 16557.9},
    {"Dataset": "BVP", "Size": 7.3e8, "Algorithm": "GPU (OpenCL)", "Time": 1583.48},
]

df = pd.DataFrame(data)

# ================= VYKRESLENÍ =================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))

# Barvy: Šedá (Seq), Modrá (Par), Zelená (Par+SIMD), Červená (GPU)
palette = {
    "CPU Sequential": "gray",
    "CPU Parallel (No SIMD)": "#3498db", # Modrá
    "CPU Parallel + SIMD": "#2ecc71",    # Zelená
    "GPU (OpenCL)": "#e74c3c"            # Červená
}

markers = {
    "CPU Sequential": "o",
    "CPU Parallel (No SIMD)": "^",
    "CPU Parallel + SIMD": "s",
    "GPU (OpenCL)": "D"
}

sns.lineplot(
    data=df,
    x="Size",
    y="Time",
    hue="Algorithm",
    style="Algorithm",
    markers=markers,
    dashes=False,
    palette=palette,
    linewidth=2.5,
    markersize=9
)

# --- NASTAVENÍ OS ---
plt.xscale("log")    # Osa X logaritmická (velikost vstupu)
plt.yscale("linear") # Osa Y lineární (pro zdůraznění rozdílu času)

# Titulky
plt.title("Kompletní porovnání škálovatelnosti", fontsize=16, pad=20)
plt.xlabel("Počet zpracovávaných prvků (Log scale)", fontsize=12)
plt.ylabel("Čas výpočtu v ms (Linear scale)", fontsize=12)

# Vlastní popisky osy X
plt.xticks(
    [3.6e4, 1.1e7, 7.3e8], 
    ['Dexcom\n(37k)', 'HR\n(11M)', 'BVP\n(740M)'], 
    fontsize=11
)

# Anotace
# 1. GPU vs CPU
plt.annotate('GPU (1.5s) vs CPU Seq (81s)', 
             xy=(7.3e8, 1583), 
             xytext=(5e7, 40000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, fontweight='bold')

# 2. Vliv SIMD
plt.annotate('Vliv SIMD (4x zrychlení)', 
             xy=(7.3e8, 16557), 
             xytext=(2e7, 15000),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=8),
             fontsize=10, color='green', fontweight='bold')

plt.grid(True, which="major", ls="-", alpha=0.5)
plt.legend(title="Algoritmus", loc="upper left")

plt.tight_layout()
plt.savefig("scalability_complete.png", dpi=300)
print("Graf uložen jako scalability_complete.png")
plt.show()
