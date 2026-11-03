"""
plot_image_vs_metadata.py
─────────────────────────
Generates Figure 9: Side-by-Side Subgroup Comparison between
Image-Only 3D ResNet (n_meta=0) and Metadata-Assisted 3D ResNet (n_meta=3).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
COMP_CSV = BASE_DIR / "outputs" / "comparison" / "image_only_vs_metadata_comparison.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(COMP_CSV)

# Subgroups to plot
groups = ["All Subjects", "Female", "Male", "Guy's", "Hammersmith", "KCL", "WashU"]
sub_df = df[df["Group"].isin(groups)].copy()
# Order correctly
sub_df["Group"] = pd.Categorical(sub_df["Group"], categories=groups, ordered=True)
sub_df = sub_df.sort_values("Group")

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

bars1 = ax.bar(x - width/2, sub_df["Image_Only_MAE"], width, label="Image-Only (n_meta=0)", color="#3b82f6", alpha=0.9, edgecolor="black", linewidth=0.8)
bars2 = ax.bar(x + width/2, sub_df["Metadata_MAE"], width, label="Metadata-Assisted (n_meta=3)", color="#f59e0b", alpha=0.9, edgecolor="black", linewidth=0.8)

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel("Mean Absolute Error (years)", fontsize=11, fontweight='bold')
ax.set_title("Subgroup Performance: Image-Only vs. Metadata-Assisted 3D ResNet", fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10, fontweight='bold')
ax.set_ylim(0, 10.5)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

plt.tight_layout()
out_path = FIG_DIR / "fig9_image_vs_metadata.png"
plt.savefig(out_path)
plt.close()
print(f"Saved Figure 9 to {out_path}")

