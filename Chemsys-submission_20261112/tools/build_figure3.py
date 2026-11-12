"""Render the primary-model and DeepSeek rows of Figure 3."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1] / "paper/hypothesis_scope_jcim"
SOURCE = PAPER / "figure_data" / "table4_deepseek"
ARMS = (
    ("r1", "Controller", "#0072B2", "-"),
    ("rbo", "Reasoning BO", "#CC79A7", "--"),
    ("bo", "Random-start BO", "#222222", "-."),
    ("lgbo", "LGBO", "#E69F00", ":"),
)
MODELS = (("original", "Primary LLM (gpt-6.1-terra)"),
          ("deepseek", "DeepSeek (deepseek-flash)"))
SEED = 20261106
T90_DF23 = 1.714  # Same rounded critical value used in the existing figure.


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_and_check():
    records = rows(SOURCE / "trajectories.csv")
    assert len(records) == 5760
    grouped = defaultdict(dict)
    for row in records:
        key = row["model"], row["task"], row["method"], int(row["pair"])
        query = int(row["query"])
        assert query not in grouped[key]
        grouped[key][query] = row
    best = defaultdict(list)
    pair_ids = {}
    for (model, task, method, pair), run in sorted(grouped.items()):
        assert set(run) == set(range(1, 21))
        measured = np.array([float(run[q]["value_pct_of_max"]) for q in range(1, 21)])
        running = np.array([float(run[q]["running_best_pct_of_max"]) for q in range(1, 21)])
        np.testing.assert_allclose(running, np.maximum.accumulate(measured), rtol=0, atol=1e-10)
        best[model, task, method].append(running)
        pair_ids.setdefault((model, task, method), []).append(pair)
    best = {key: np.asarray(value) for key, value in best.items()}
    assert len(best) == 12
    for key, value in best.items():
        assert value.shape == (24, 20)
        assert pair_ids[key] == list(range(24))
        model, task, method = key
        np.testing.assert_allclose(value[:, 0], best[model, task, "bo"][:, 0], rtol=0, atol=1e-10)

    paired = rows(SOURCE / "paired_results.csv")
    assert len(paired) == 288
    for row in paired:
        key = row["model"], row["task"], row["method"]
        i = pair_ids[key].index(int(row["pair"]))
        auc = best[key][i].mean()
        bo = best[key[0], key[1], "bo"][i].mean()
        np.testing.assert_allclose(
            [auc, best[key][i, -1], bo, auc - bo],
            [float(row[name]) for name in ("SearchAUC", "final_best", "BO_SearchAUC", "paired_difference")],
            rtol=0, atol=1e-10,
        )
    summaries = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))["summary"]
    for row in summaries:
        key = row["model"], row["task"], row["method"]
        values = best[key]
        delta = values.mean(axis=1) - best[key[0], key[1], "bo"].mean(axis=1)
        half = T90_DF23 * delta.std(ddof=1) / np.sqrt(24)
        np.testing.assert_allclose(
            [values.mean(), values[:, -1].mean(), delta.mean(), delta.mean() - half, delta.mean() + half],
            [float(row[name]) for name in ("SearchAUC", "final_best", "paired_difference", "interval90_low", "interval90_high")],
            rtol=0, atol=1e-10,
        )
    return best


def _render(out):
    derived = out / "fig3_models"
    best = load_and_check()
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 6.3, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
        "pdf.fonttype": 42, "lines.linewidth": 1.4,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.65), gridspec_kw={
        "width_ratios": [1.3, 1.05, .62], "wspace": .47, "hspace": .88,
        "left": .09, "right": .985, "bottom": .085, "top": .895,
    })
    curve_rows, pair_rows, summary_rows = [], [], []
    for row_index, (model, title) in enumerate(MODELS):
        rng = np.random.default_rng(SEED)
        a, b, c = axes[row_index]
        fig.text(.09, a.get_position().y1 + .073, title, fontsize=9, fontweight="bold", va="bottom")
        for arm, label, color, style in ARMS:
            values = best[model, "aryl", arm]
            bounds = np.array([
                np.quantile(values[:, i][rng.integers(0, 24, (10000, 24))].mean(axis=1), [.05, .95])
                for i in range(20)
            ])
            mean = values.mean(axis=0)
            a.fill_between(range(1, 21), bounds[:, 0], bounds[:, 1], color=color, alpha=.16, lw=0)
            a.plot(range(1, 21), mean, color=color, ls=style, label=label)
            for i in range(20):
                curve_rows.append(dict(model=model, task="aryl", method=arm, query=i + 1,
                                       mean=mean[i], bootstrap90_low=bounds[i, 0], bootstrap90_high=bounds[i, 1]))
        a.set(xlim=(1, 20), ylim=(0, 102), xticks=[1, 5, 10, 15, 20], xlabel="Query",
              ylabel="Mean running best (% of maximum)")
        a.legend(loc="lower right", handlelength=1.8)

        for ax, task, arms in ((b, "aryl", [ARMS[0], ARMS[1], ARMS[3]]), (c, "cpa", [ARMS[0]])):
            for i, (arm, label, color, style) in enumerate(arms):
                delta = best[model, task, arm].mean(axis=1) - best[model, task, "bo"].mean(axis=1)
                half = T90_DF23 * delta.std(ddof=1) / np.sqrt(24)
                ax.scatter(i + rng.uniform(-.17, .17, len(delta)), delta, s=9, color=color, alpha=.6, lw=0)
                ax.errorbar(i + .32, delta.mean(), yerr=half, fmt="o", color="#222222", ms=3.8, capsize=2, lw=1.1)
                pair_rows.extend(dict(model=model, task=task, method=arm, pair=pair, difference=value)
                                 for pair, value in enumerate(delta))
                summary_rows.append(dict(model=model, task=task, method=arm, pairs=24,
                                         mean_difference=delta.mean(), interval90_low=delta.mean() - half,
                                         interval90_high=delta.mean() + half))
            ax.axhline(0, color="#8C8C8C", lw=.8, ls="--")
            ax.set_xticks(range(len(arms)), [label.replace("Reasoning BO", "Reasoning\nBO")
                                           for _, label, _, _ in arms], fontsize=6.8)
            ax.set(xlim=(-.5, len(arms) - .4), ylim=(-58, 58), yticks=[-50, -25, 0, 25, 50])
        b.set_ylabel("Minus random-start BO, same pair\n(SearchAUC, % of maximum)")
        c.tick_params(labelleft=False)
        for col, ax in enumerate((a, b, c)):
            ax.set_title("Direct Arylation" if col < 2 else "CPA", loc="left")
            ax.text(-.02, 1.04, "abcdef"[row_index * 3 + col], transform=ax.transAxes,
                    fontsize=9.5, fontweight="bold", ha="right", va="bottom")

    derived.mkdir(parents=True, exist_ok=True)
    write_csv(derived / "mean_running_best.csv", curve_rows)
    write_csv(derived / "paired_differences.csv", pair_rows)
    write_csv(derived / "difference_summary.csv", summary_rows)
    provenance = {
        "sources_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                           for name in ("trajectories.csv", "paired_results.csv", "results.json")},
        "layout": "primary model a-c; DeepSeek d-f; common corresponding axes",
        "pairs_per_model_and_task": 24, "queries_per_trajectory": 20,
        "normalization": "percent of task pool maximum, unchanged from frozen records",
        "SearchAUC": "mean of the 20 post-query running maxima, including the shared start",
        "curve_intervals": "pointwise 90% percentile bootstrap over pairs; 10000 resamples per query",
        "bootstrap_seed_per_model": SEED,
        "difference_intervals": "paired 90% t interval; df=23; original critical value=1.714",
        "BO_reference": "corresponding model, task and pair, as in Table 4",
        "source_checks": "5760 observations, 288 paired summaries and all 12 Table 4 rows agree",
    }
    (derived / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    target = out / "fig3_heldout_shared.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print("PASS Figure 3 and its 288 paired summary rows")


def render(out):
    with plt.rc_context():
        _render(out)
