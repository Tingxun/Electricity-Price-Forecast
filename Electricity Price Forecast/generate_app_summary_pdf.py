from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "app_summary_one_page.pdf"


TITLE = "Electricity Price Forecast Repo Summary"

SECTIONS = [
    (
        "What it is",
        [
            "A local Python-based workflow for Hubei electricity price forecasting research and exploratory analysis.",
            "Repo evidence shows a data aggregation script, an analysis notebook, local datasets, charts, JSON results, and markdown reports rather than a deployed app or service.",
        ],
    ),
    (
        "Who it's for",
        [
            "Primary persona: an analyst or researcher studying day-ahead electricity prices with Hubei market, load, renewable, hydro, tie-line, and weather data.",
        ],
    ),
    (
        "What it does",
        [
            "Merges market-boundary inputs with average clearing price data into one hourly CSV dataset.",
            "Cleans date fields, aligns records by date + time period, and adds calendar/hour features.",
            "Loads the merged CSV into a notebook for exploratory data analysis and statistics.",
            "Computes summary metrics for price and load, including time range, record count, and correlations.",
            "Produces visual analysis charts under the local chart output folder.",
            "Stores analysis outputs as JSON results and markdown reports.",
        ],
    ),
    (
        "How it works",
        [
            "Input data: Excel workbooks under the local Hubei dataset folder, including market-boundary files and city weather files.",
            "Processing: data_aggregation.py reads market-boundary and clearing-price workbooks, converts dates, merges on a timestamp key, adds time features, and writes the merged CSV.",
            "Analysis: data_analysis.ipynb reads that CSV with pandas/numpy and uses matplotlib/seaborn for EDA, correlation checks, and chart generation.",
            "Outputs: data_analysis_results.json, PNG charts in the chart folder, and markdown reports that document findings and experiment design.",
            "Service/API layer: Not found in repo.",
        ],
    ),
    (
        "How to run",
        [
            "Use Python with pandas, numpy, matplotlib, and seaborn available.",
            "Keep the local dataset folder in the repo root so the script paths resolve.",
            "Run: python data_aggregation.py",
            "Open and run: data_analysis.ipynb",
            "Requirements file / one-command install / app server startup: Not found in repo.",
        ],
    ),
]

FOOTER = (
    "Evidence used: the aggregation script, analysis notebook, JSON results, repo markdown docs, "
    "the local Hubei dataset folder, and the generated chart folder."
)


def wrap_paragraph(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


def build_lines() -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for heading, items in SECTIONS:
        lines.append(("heading", heading))
        for item in items:
            wrapped = wrap_paragraph(item, 88)
            if len(wrapped) == 1:
                lines.append(("bullet", f"- {wrapped[0]}"))
                continue
            lines.append(("bullet", f"- {wrapped[0]}"))
            for extra in wrapped[1:]:
                lines.append(("cont", f"  {extra}"))
        lines.append(("space", ""))
    lines.append(("footer", FOOTER))
    return lines


def render_pdf(output_path: Path) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(
        0.06,
        0.965,
        TITLE,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
        family="DejaVu Sans",
    )

    fig.text(
        0.06,
        0.942,
        "Single-page summary based only on repository evidence",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#4e6478",
        family="DejaVu Sans",
    )

    y = 0.915
    line_gap = 0.0192
    heading_gap = 0.024

    for kind, text in build_lines():
        if kind == "heading":
            y -= 0.004
            fig.text(
                0.06,
                y,
                text,
                ha="left",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                color="#17324d",
                family="DejaVu Sans",
            )
            y -= heading_gap
        elif kind == "bullet":
            fig.text(
                0.075,
                y,
                text,
                ha="left",
                va="top",
                fontsize=9.25,
                color="#1d1d1f",
                family="DejaVu Sans",
            )
            y -= line_gap
        elif kind == "cont":
            fig.text(
                0.092,
                y,
                text,
                ha="left",
                va="top",
                fontsize=9.1,
                color="#1d1d1f",
                family="DejaVu Sans",
            )
            y -= line_gap
        elif kind == "space":
            y -= 0.009
        elif kind == "footer":
            wrapped_footer = wrap_paragraph(text, 108)
            for footer_line in wrapped_footer:
                fig.text(
                    0.06,
                    y,
                    footer_line,
                    ha="left",
                    va="top",
                    fontsize=7.7,
                    color="#5f6f7f",
                    family="DejaVu Sans",
                )
                y -= 0.0145

    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    render_pdf(OUTPUT_PATH)
    print(f"Created PDF: {OUTPUT_PATH}")
