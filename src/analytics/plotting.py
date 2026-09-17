import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CONFIG, resolve

OUTPUT = resolve(CONFIG["analytics"]["output"])
DPI = CONFIG["analytics"]["dpi"]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
YELLOW = "#eda100"
RED = "#e34948"
LIGHT_BLUE = "#cde2fb"
GRAY = "#8a8985"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})


def save(fig, section, name):
    folder = OUTPUT / section
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=DPI)
    plt.close(fig)


def label_bars(ax, bars, labels, horizontal=False, inside=False):
    color = SURFACE if inside else TEXT_SECONDARY
    for bar, text in zip(bars, labels):
        if horizontal:
            xy = (bar.get_width(), bar.get_y() + bar.get_height() / 2)
            offset, va, ha = ((-4, 0), "center", "right") if inside else ((4, 0), "center", "left")
        else:
            xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
            offset, va, ha = ((0, -4), "top", "center") if inside else ((0, 3), "bottom", "center")
        ax.annotate(text, xy, xytext=offset, textcoords="offset points",
                    va=va, ha=ha, fontsize=8, color=color)
