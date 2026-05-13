import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

modelos = ["Baseline\n(number-plate-yolo26s)", "Fine-tuning v1\n(hiperparâmetros manuais)", "Fine-tuning v2\n(tune genético)"]
carros  = [70.30, 86.81, 89.18]
motos   = [61.78, 88.24, 91.99]
geral   = [68.60, 87.10, 89.74]

x = np.arange(len(modelos))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#0f0f0f")

bars1 = ax.bar(x - width, carros, width, label="Carros", color="#4e9af1", zorder=3)
bars2 = ax.bar(x,         motos,  width, label="Motos",  color="#f1a84e", zorder=3)
bars3 = ax.bar(x + width, geral,  width, label="Geral",  color="#6ef14e", zorder=3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")

ax.set_ylim(50, 100)
ax.set_ylabel("IoU Médio (%)", color="white", fontsize=12)
ax.set_title("Evolução do IoU — Detecção de Placas Veiculares BR", color="white", fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(modelos, color="white", fontsize=10)
ax.tick_params(colors="white")
ax.yaxis.set_tick_params(labelcolor="white")
ax.grid(axis="y", color="#333333", zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=10)

plt.tight_layout()
plt.savefig("evolucao_iou.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Imagem salva: evolucao_iou.png")
