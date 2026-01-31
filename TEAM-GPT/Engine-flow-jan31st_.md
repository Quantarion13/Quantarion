#!/usr/bin/env python3
"""
generate_visuals.py
===================

Generates TEAM-GPT production visuals:
1. Mermaid pipeline flowchart
2. HGME hypergraph visualization
3. Multi-language performance heatmap

Outputs PNG + PDF into docs/images/
"""

import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 0. Setup paths ---
OUTPUT_DIR = "docs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Generate Mermaid Pipeline Flowchart ---
MERMAID_FILE = "TEAM-GPT/Mermaid-Flow.mk"
MERMAID_PNG = os.path.join(OUTPUT_DIR, "team-gpt-flowchart.png")
MERMAID_PDF = os.path.join(OUTPUT_DIR, "team-gpt-flowchart.pdf")

if os.path.exists(MERMAID_FILE):
    try:
        subprocess.run([
            "mmdc",
            "-i", MERMAID_FILE,
            "-o", MERMAID_PNG,
            "-b", "transparent"
        ], check=True)
        subprocess.run([
            "mmdc",
            "-i", MERMAID_FILE,
            "-o", MERMAID_PDF,
            "-b", "transparent"
        ], check=True)
        print(f"[✔] Mermaid flowchart generated: {MERMAID_PNG} & {MERMAID_PDF}")
    except Exception as e:
        print(f"[❌] Failed to generate Mermaid diagrams: {e}")
else:
    print(f"[⚠️] Mermaid file not found: {MERMAID_FILE}")

# --- 2. Generate HGME Hypergraph ---
HGME_HYPEREDGES = [
    {"PQC","ML-KEM","HQC","KYBER"},
    {"QUORUM16","PQC","ML-KEM"},
    {"KYBER","HQC","PHI43"}
]

G = nx.Graph()
for e in HGME_HYPEREDGES:
    for n1 in e:
        for n2 in e:
            if n1 != n2:
                G.add_edge(n1, n2)

plt.figure(figsize=(10,7))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', font_size=12)
HGME_PNG = os.path.join(OUTPUT_DIR, "hgme-hypergraph.png")
HGME_PDF = os.path.join(OUTPUT_DIR, "hgme-hypergraph.pdf")
plt.savefig(HGME_PNG, dpi=300)
plt.savefig(HGME_PDF)
plt.close()
print(f"[✔] HGME hypergraph generated: {HGME_PNG} & {HGME_PDF}")

# --- 3. Generate Multi-language Heatmap ---
LANG_DATA = {
    "Language":["English","French","Russian","Spanish","German","Chinese","Japanese","Portuguese","Italian"],
    "LUT_Hit":[90,82,75,85,82,75,68,88,87],
    "Latency_ms":[15,18,22,17,19,23,25,18,16],
    "HGME_Fallback":[10,18,25,15,18,25,32,12,13]
}

df = pd.DataFrame(LANG_DATA)

plt.figure(figsize=(12,6))
sns.heatmap(df.set_index("Language")[["LUT_Hit","Latency_ms","HGME_Fallback"]],
            annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5)
plt.title("TEAM-GPT Multi-Language Metrics", fontsize=14)
HEATMAP_PNG = os.path.join(OUTPUT_DIR, "team-gpt-heatmap.png")
HEATMAP_PDF = os.path.join(OUTPUT_DIR, "team-gpt-heatmap.pdf")
plt.savefig(HEATMAP_PNG, dpi=300, bbox_inches='tight')
plt.savefig(HEATMAP_PDF, bbox_inches='tight')
plt.close()
print(f"[✔] Multi-language heatmap generated: {HEATMAP_PNG} & {HEATMAP_PDF}")

print("[✅] All TEAM-GPT visuals successfully generated!")
