"""Generate the benchmark charts embedded in the Chinese README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets" / "sm89"
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
)

COLORS = {
    "ink": "#17212B",
    "muted": "#5E6A75",
    "grid": "#DDE3E8",
    "paper": "#F5F7F9",
    "white": "#FFFFFF",
    "blue": "#276FBF",
    "green": "#2F855A",
    "teal": "#167D8D",
    "coral": "#D95F45",
    "gold": "#C58B20",
}

C1_DECODE = 54.75
C1_E2E = 50.36
C8_SCHEDULER = (382.8 + 383.8) / 2
C8_DECODE_WINDOW = 374.58
C8_E2E = 318.59
GPU_UTILIZATION = 92.7


def configure_font() -> None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=candidate
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 13


def style_axis(ax, xmax: float) -> None:
    ax.set_facecolor(COLORS["white"])
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors=COLORS["muted"], length=0)


def label_bars(ax, bars, labels, xmax: float) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=COLORS["ink"],
            fontsize=14,
            fontweight="bold",
        )


def save_throughput_chart() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=COLORS["paper"])
    fig.subplots_adjust(left=0.11, right=0.95, top=0.76, bottom=0.16, wspace=0.38)
    fig.suptitle(
        "8×RTX 4090 推理吞吐实测",
        x=0.08,
        y=0.93,
        ha="left",
        fontsize=28,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.08,
        0.865,
        "DeepSeek-V4-Flash-0731 MXFP4 · TP=8 · 256 输入 token / 128 输出 token",
        ha="left",
        fontsize=14,
        color=COLORS["muted"],
    )

    c1_labels = ["端到端", "解码"]
    c1_values = [C1_E2E, C1_DECODE]
    bars = axes[0].barh(
        c1_labels,
        c1_values,
        height=0.48,
        color=[COLORS["green"], COLORS["blue"]],
    )
    style_axis(axes[0], 65)
    axes[0].set_title("C1 单请求", loc="left", fontsize=19, fontweight="bold")
    axes[0].set_xlabel("token/s", color=COLORS["muted"])
    label_bars(axes[0], bars, ["50.36", "54.75"], 65)

    c8_labels = ["端到端聚合", "解码窗口聚合", "Scheduler 输出"]
    c8_values = [C8_E2E, C8_DECODE_WINDOW, C8_SCHEDULER]
    bars = axes[1].barh(
        c8_labels,
        c8_values,
        height=0.48,
        color=[COLORS["green"], COLORS["teal"], COLORS["coral"]],
    )
    style_axis(axes[1], 430)
    axes[1].set_title("C8 并发聚合", loc="left", fontsize=19, fontweight="bold")
    axes[1].set_xlabel("token/s", color=COLORS["muted"])
    label_bars(axes[1], bars, ["318.59", "374.58", "382.8–383.8"], 430)

    fig.text(
        0.08,
        0.07,
        "注：C1 与 C8 使用不同并发口径；Scheduler、解码窗口和端到端吞吐不可互相替代。",
        ha="left",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.savefig(
        OUTPUT_DIR / "benchmark-throughput.png", dpi=160, facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def save_efficiency_chart() -> None:
    ideal_c8 = C1_DECODE * 8
    linear_efficiency = [
        C8_SCHEDULER / ideal_c8 * 100,
        C8_DECODE_WINDOW / ideal_c8 * 100,
        C8_E2E / ideal_c8 * 100,
    ]
    stage_retention = [
        C8_DECODE_WINDOW / C8_SCHEDULER * 100,
        C8_E2E / C8_SCHEDULER * 100,
        GPU_UTILIZATION,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=COLORS["paper"])
    fig.subplots_adjust(left=0.13, right=0.95, top=0.76, bottom=0.16, wspace=0.42)
    fig.suptitle(
        "TP8 扩展效率与吞吐保留率",
        x=0.08,
        y=0.93,
        ha="left",
        fontsize=28,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.08,
        0.865,
        f"理想线性基准：C1 解码 54.75 × 8 = {ideal_c8:.2f} token/s",
        ha="left",
        fontsize=14,
        color=COLORS["muted"],
    )

    labels = ["端到端 / 理想值", "解码窗口 / 理想值", "Scheduler / 理想值"]
    bars = axes[0].barh(
        labels,
        linear_efficiency[::-1],
        height=0.48,
        color=[COLORS["green"], COLORS["teal"], COLORS["coral"]],
    )
    style_axis(axes[0], 105)
    axes[0].axvline(100, color=COLORS["ink"], linewidth=1.5, linestyle="--")
    axes[0].set_title("相对线性扩展", loc="left", fontsize=19, fontweight="bold")
    axes[0].set_xlabel("效率（%）", color=COLORS["muted"])
    label_bars(
        axes[0], bars, [f"{value:.1f}%" for value in linear_efficiency[::-1]], 105
    )

    labels = ["GPU 平均利用率", "端到端 / Scheduler", "解码窗口 / Scheduler"]
    bars = axes[1].barh(
        labels,
        stage_retention[::-1],
        height=0.48,
        color=[COLORS["teal"], COLORS["green"], COLORS["gold"]],
    )
    style_axis(axes[1], 105)
    axes[1].axvline(100, color=COLORS["ink"], linewidth=1.5, linestyle="--")
    axes[1].set_title("C8 阶段保留率", loc="left", fontsize=19, fontweight="bold")
    axes[1].set_xlabel("比例（%）", color=COLORS["muted"])
    label_bars(axes[1], bars, [f"{value:.1f}%" for value in stage_retention[::-1]], 105)

    fig.text(
        0.08,
        0.07,
        "效率仅描述本次固定硬件与请求配置；它不是跨模型、跨引擎的通用缩放结论。",
        ha="left",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.savefig(
        OUTPUT_DIR / "parallel-efficiency.png", dpi=160, facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_font()
    save_throughput_chart()
    save_efficiency_chart()
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"generated {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
