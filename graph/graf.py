import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

# ================= NASTAVENÍ =================
INPUT_FILE = "benchmark_logs.txt"
OUTPUT_FILE = "speedup_graph.png"

# Pokud nemáš soubor, vytvoříme si ho z tvých dat pro demonstraci
raw_data = """
DEXCOM >> [BENCHMARK] Read of DEXCOM data: 166.129 ms
DEXCOM >> [BENCHMARK] CPU Sequential: 5.092 ms
DEXCOM >> [BENCHMARK] CPU Parallel + Vectorized: 0.875167 ms
DEXCOM >> [BENCHMARK] CPU Parallel: 4.60637 ms
DEXCOM >> [BENCHMARK] GPU: 127.735 ms
HR >> [BENCHMARK] Read of HR data: 18435.5 ms
HR >> [BENCHMARK] CPU Sequential: 1573.03 ms
HR >> [BENCHMARK] CPU Parallel + Vectorized: 230.387 ms
HR >> [BENCHMARK] CPU Parallel - NoVect: 1396.68 ms
HR >> [BENCHMARK] GPU: 19.6603 ms
BVP >> [BENCHMARK] Read of BVP data: 1.52389e+06 ms
BVP >> [BENCHMARK] CPU Sequential: 81742.2 ms
BVP >> [BENCHMARK] CPU Parallel + Vectorized: 16557.9 ms
BVP >> [BENCHMARK] CPU Parallel - NoVect: 68383.3 ms
BVP >> [BENCHMARK] GPU - kernel.cl: 1583.48 ms
"""

# Funkce pro parsování řádků
def parse_logs(content):
    data = []
    # Regex pro extrakci: DATASET >> [BENCHMARK] ALGO: TIME ms
    pattern = re.compile(r"(\w+) >> \[BENCHMARK\] (.*?): ([\d\.e\+\-]+) ms")
    
    for line in content.strip().split('\n'):
        match = pattern.search(line)
        if match:
            dataset, algo, time_str = match.groups()
            
            # Ignorujeme řádky s načítáním dat (Read of...)
            if "Read of" in algo:
                continue
                
            # Normalizace názvů algoritmů (aby byly v grafu hezké a jednotné)
            algo = algo.strip()
            if "Sequential" in algo:
                clean_algo = "Sequential (Baseline)"
            elif "Vectorized" in algo:
                clean_algo = "Parallel + SIMD"
            elif "NoVect" in algo or algo == "CPU Parallel":
                clean_algo = "Parallel (No SIMD)"
            elif "GPU" in algo:
                clean_algo = "GPU (OpenCL)"
            else:
                clean_algo = algo # Fallback

            data.append({
                "Dataset": dataset,
                "Algorithm": clean_algo,
                "Time": float(time_str)
            })
    return pd.DataFrame(data)

# ================= ZPRACOVÁNÍ DAT =================

# Zkusíme načíst soubor, jinak použijeme ukázková data
try:
    with open(INPUT_FILE, "r") as f:
        df = parse_logs(f.read())
    print(f"Načteno ze souboru {INPUT_FILE}")
except FileNotFoundError:
    print("Soubor nenalezen, používám demo data...")
    df = parse_logs(raw_data)

# Výpočet Speedupu
# 1. Najdeme referenční časy (Sequential) pro každý dataset
baseline = df[df["Algorithm"] == "Sequential (Baseline)"].set_index("Dataset")["Time"]

# 2. Funkce pro výpočet speedupu
def calculate_speedup(row):
    seq_time = baseline.get(row["Dataset"])
    if seq_time:
        return seq_time / row["Time"]
    return 0

df["Speedup"] = df.apply(calculate_speedup, axis=1)

# ================= VYKRESLENÍ GRAFU =================

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Definice pořadí algoritmů v legendě
algo_order = ["Sequential (Baseline)", "Parallel (No SIMD)", "Parallel + SIMD", "GPU (OpenCL)"]

# Vykreslení
chart = sns.barplot(
    data=df,
    x="Dataset",
    y="Speedup",
    hue="Algorithm",
    hue_order=algo_order,
    palette="viridis",
    edgecolor="black"
)

# Úpravy grafu
plt.title("Zrychlení algoritmů oproti sekvenční verzi (Speedup)", fontsize=16, pad=20)
plt.ylabel("Zrychlení (x-krát)", fontsize=12)
plt.xlabel("Datová sada", fontsize=12)

# Logaritmická škála na Y je nutná, protože rozdíly jsou extrémní 
# (Dexcom GPU je 0.04x, zatímco BVP GPU je 50x)
plt.yscale("log")
plt.grid(True, which="minor", ls="--", alpha=0.3)

# Přidání popisků hodnot nad sloupce
for container in chart.containers:
    # Formátování popisku: pokud je > 1, formát "51.6x", pokud < 1, formát "0.04x"
    labels = [f"{val:.2f}×" if val < 10 else f"{val:.1f}×" for val in container.datavalues]
    chart.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight="bold")

# Legenda
plt.legend(title="Metoda", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Graf uložen jako {OUTPUT_FILE}")
plt.show()
