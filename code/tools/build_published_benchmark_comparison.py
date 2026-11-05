"""Read published tables and completed local summaries; no experimental runs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/reports/published_benchmark_comparison_20261105"
SOURCE = OUT / "sources"
FGSS_URL = "https://arxiv.org/html/2610.21246v1#A8"
METHODS = ("Random Search", "HEBO", "GP-EI", "LLAMBO-adapted", "FGSS-BO")

# Appendix A.8, Table 10. Columns follow METHODS; tuple = (mean, sample SD).
PUBLISHED = {
    "c1": dict(SearchAUC=[(64.033,9.672),(74.673,9.317),(83.315,5.291),(55.087,13.751),(65.738,15.484)],
               final_best=[(67.505,10.443),(86.480,6.887),(91.424,.019),(60.071,15.792),(69.751,14.764)]),
    "c2": dict(SearchAUC=[(19.916,6.719),(32.166,6.307),(32.762,4.777),(19.164,6.721),(21.262,8.662)],
               final_best=[(21.602,6.882),(37.151,6.817),(40.085,0),(20.910,7.124),(22.286,9.016)]),
    "c3": dict(SearchAUC=[(72.094,3.800),(85.115,3.114),(84.723,4.589),(68.152,3.466),(71.726,2.183)],
               final_best=[(79.543,5.887),(89.536,2.604),(92.770,.241),(73.964,8.452),(76.606,4.921)]),
    "c4": dict(SearchAUC=[(88.079,2.992),(93.910,2.093),(94.243,1.450),(89.939,2.651),(88.505,3.597)],
               final_best=[(90.216,3.366),(95.541,2.615),(96.688,.034),(92.000,4.124),(89.866,3.561)]),
    "snar": dict(SearchAUC=[(.572,.078),(.967,.048),(.990,.049),(.579,.101),(.573,.123)],
                 final_best=[(.593,.074),(1.051,.005),(1.058,.001),(.641,.107),(.680,.057)]),
}


def save_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paper = (SOURCE / "fgss.html").read_text(encoding="utf-8")
    metrics = []
    for task, measures in PUBLISHED.items():
        for metric, entries in measures.items():
            for method, (mean, sd) in zip(METHODS, entries):
                # Verify the transcribed numeric literals occur in the frozen original.
                assert f"{mean:.3f}" in paper and f"{sd:.3f}" in paper, (task, metric, method)
                metrics.append(dict(paper="FGSS-BO", version="2610.21246v1", locator="Appendix A.8 Table 10",
                    task=task, method=method, metric=metric, mean=mean, sample_SD=sd, n_runs=6,
                    initialization=8, search_evaluations=20,
                    objective="STY/10000-E_factor/100" if task=="snar" else "emulator yield percent",
                    source_url=FGSS_URL,
                    comparability="different objective" if task=="snar" else "same benchmark family; continuous search, 8+20 vs fixed pool, 0+20"))
    for method, auc, auc_sd, final, final_sd in (
        ("LLAMBO-adapted",1677.365,465.783,1803.720,521.827),
        ("FGSS-BO",1765.846,381.965,1962.651,237.131)):
        for metric, mean, sd in (("SearchAUC",auc,auc_sd),("final_best",final,final_sd)):
            assert f"{mean:.3f}" in paper and f"{sd:.3f}" in paper
            metrics.append(dict(paper="FGSS-BO",version="2610.21246v1",locator="Appendix A.10",
                task="hplc",method=method,metric=metric,mean=mean,sample_SD=sd,n_runs=6,
                initialization="not separately specified in A.10",search_evaluations="not separately specified in A.10",
                objective="HPLC peak response",source_url="https://arxiv.org/html/2610.21246v1#A10",
                comparability="same benchmark family; archived lookup/domain and initialization are not established identical"))
    save_csv("published_search_metrics.csv", metrics)

    with (ROOT / "experiments/reports/hypothesis_scope_vs_reasoning_bo_20261105/comparison.csv").open(encoding="utf-8",newline="") as stream:
        current = {r["task"]:r for r in csv.DictReader(stream)}
    side = []
    for task in ("c1","c2","c3","c4","hplc"):
        row = dict(task=task, current_initial=0,current_queries=20,current_n=4,
            current_SearchAUC_mean=float(current[task]["current_hypothesis_SearchAUC_mean"]),
            current_SearchAUC_SD=float(current[task]["current_hypothesis_SearchAUC_sample_sd"]),
            current_final_best_mean=float(current[task]["current_hypothesis_final_best_mean"]),
            current_final_best_SD=float(current[task]["current_hypothesis_final_best_sample_sd"]))
        for method, name in (("LLAMBO-adapted","published_LLAMBO"),("FGSS-BO","published_FGSS"),("GP-EI","published_GP_EI"),("HEBO","published_HEBO")):
            for metric in ("SearchAUC","final_best"):
                hit = [r for r in metrics if (r["task"],r["method"],r["metric"])==(task,method,metric)]
                row[f"{name}_{metric}_mean"]=hit[0]["mean"] if hit else ""
                row[f"{name}_{metric}_SD"]=hit[0]["sample_SD"] if hit else ""
        row["strict_matched_comparison"]=False
        side.append(row)
    save_csv("side_by_side.csv", side)

    # Read archived response logs only to audit completeness; do not compute a
    # valid-only SearchAUC that would silently exclude failed proposals.
    archived_audit = []
    for filename in ("coscientist_suzuki_no_prior.csv","coscientist_suzuki_prior.csv","coscientist_bh.csv"):
        groups=defaultdict(list)
        with (SOURCE/filename).open(encoding="utf-8-sig",newline="") as stream:
            for r in csv.DictReader(stream):
                key=tuple((k,r[k]) for k in r if k not in ("message_id","role","message"))
                groups[key].append(r)
        counts=Counter()
        for group in groups.values():
            observed=sum(bool(re.fullmatch(r"The yield of this reaction is ([-+\d.]+)%",r["message"].strip()))
                         for r in group if r["role"]=="user")
            counts[observed]+=1
        archived_audit.append(dict(file=filename,trajectories=len(groups),returned_yield_counts=dict(sorted(counts.items())),
            inference="Not a published SearchAUC table. Failed/invalid proposal accounting and unequal strata prevent a naive valid-only pooled comparison."))
    (OUT/"coscientist_archive_audit.json").write_text(json.dumps(archived_audit,indent=2)+"\n",encoding="utf-8")

    sources = [
        dict(id="LLAMBO",title="Large Language Models to Enhance Bayesian Optimization",year=2024,
             url="https://arxiv.org/html/2402.03921v2",scope="HPO benchmarks; no original results on these eight chemistry tasks"),
        dict(id="Coscientist",title="Autonomous chemical research with large language models",year=2023,
             url="https://www.nature.com/articles/s41586-023-06792-0",scope="Fixed substrate Perera and Doyle subproblems; 20 proposal iterations; NA and NMA"),
        dict(id="FGSS-BO",title="FGSS-BO: Feedback-Guided Hypothesis Management for LLM-Based Bayesian Optimization",year=2026,
             url="https://arxiv.org/html/2610.21246v1",scope="Preprint; Appendix A.8 Reizman1-4 and SnAr; A.10 HPLC; adapted LLAMBO comparator"),
        dict(id="Reasoning BO",title="Reasoning BO: Enhancing Bayesian Optimization With the Reasoning Power of LLMs",year=2026,
             url="https://onlinelibrary.wiley.com/doi/full/10.1002/mgea.70086",scope="Journal Table 1, with accessible arXiv2505.12833v2 Table1 and Sec4.3; batch3 and ten repeats"),
        dict(id="GOLLuM",title="Large language models as uncertainty-calibrated optimizers for experimental discovery",year=2026,
             url="https://www.nature.com/articles/s42256-026-01283-z",scope="Published 23-task study; exact supporting table values here are explicitly from arXiv2504.06265v3 (19-task version)"),
        dict(id="LGBO",title="Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery",year=2026,
             url="https://arxiv.org/html/2605.17976v1",scope="HPLC:1386 rows, interpolated continuous oracle, two initial evaluations, five seeds; Figure3 percentage-scale trace"),
        dict(id="ChemBOMAS",title="ChemBOMAS: Accelerated BO for Scientific Discovery in Chemistry with LLM-Enhanced Multi-Agent System",year=2025,
             url="https://arxiv.org/html/2509.08736",scope="5030 Suzuki samples and fixed-product BH subsets629/765; 1% labeled prior, 40 batch iterations"),
        dict(id="LLM-EI",title="Distilling and exploiting quantitative insights from large language models for enhanced Bayesian optimization of chemical reactions",year=2026,
             url="https://doi.org/10.1039/d6dd00052e",scope="Five792-row BH tasks;50 campaigns; evaluations-to-optimum; pruning schedule tuned on BH1-5"),
        dict(id="LLINBO",title="LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization",year=2025,
             url="https://arxiv.org/html/2505.14756v3",scope="Synthetic functions, HPO and3D printing; no original results on these eight tasks"),
        dict(id="BBOWP-Bench",title="BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems",year=2026,
             url="https://arxiv.org/html/2608.02612",scope="Olympus HPLC and SnAr; LLM chooses search space and optimizer; budget100, different emulator/objective"),
    ]
    (OUT/"source_registry.json").write_text(json.dumps(sources,indent=2)+"\n",encoding="utf-8")
    other=[]
    def add(paper,task,method,metric,value,spread,setup,locator,url):
        other.append(dict(paper=paper,task=task,method=method,metric=metric,value=value,reported_spread=spread,setup=setup,locator=locator,source_url=url,comparable_to_current_SearchAUC=False))
    rbo="https://arxiv.org/html/2505.12833v2"
    for method,auc,regret,imp in (("Reasoning BO",7.69,4.91,67.85),("Vanilla BO",7.59,5.42,62.32)):
        add("Reasoning BO","Perera Suzuki",method,"Log AUC",auc,"", "batch3;10 repeats; paper states30 experiments; not sequential0+20","Table1",rbo)
        add("Reasoning BO","Perera Suzuki",method,"Log Regret",regret,"", "batch3;10 repeats","Table1",rbo)
        add("Reasoning BO","Perera Suzuki",method,"IMP@1",imp,"", "paper-defined initial-performance metric","Table1",rbo)
    for method,value in (("Reasoning BO",95.06),("Vanilla BO",84.68)):
        add("Reasoning BO","Buchwald-Hartwig",method,"reported final yield percent",value,"","extended benchmark; not a matched0+20 trial","Sec4.3",rbo)
    for method,mean,sd in (("PLLMphi+T5",42.602,13.111),("GP+DRFP",37.750,13.983),("Fixed T5 embeddings",24.450,13.005)):
        add("GOLLuM preprint v3","BH1-5",method,"Top5% coverage percent",mean,sd,"10 below-median initial+50 sequential;20 seeds; SD from supporting tables","Tables3,4","https://arxiv.org/html/2504.06265v3")
    for task,control,guided in (("BH1",32,13),("BH2",22,11),("BH4 reaching99% optimum",19,9)):
        for method,value in (("EI",control),("LLM-EI",guided)):
            add("LLM-EI",task,method,"mean evaluations to target",value,"","50 campaigns;fixed-product BH subset","Sec4.2,Fig3,SI FigS5","https://doi.org/10.1039/d6dd00052e")
    for task,value,iteration in (("Suzuki",96.15,3),("Buchwald_sub1",80.00,23),("Buchwald_sub2",56.81,2)):
        add("ChemBOMAS",task,"ChemBOMAS","best found yield percent",value,"",f"1% labeled prior; batch0.1% of table;40 iterations;best iteration{iteration}","Table4","https://arxiv.org/html/2509.08736")
    for method,value in (("Gemini3Flash",2695.47),("Gemini3.1Pro",2364.52),("GPT5mini",2310.73),("GPT5.2",2246.71),("Human-selected baseline",2366.82)):
        add("BBOWP-Bench","Olympus HPLC",method,"best objective",value,"","Easy description;100 evaluations;one optimization run per designed space","Table2","https://arxiv.org/html/2608.02612")
    save_csv("other_published_metrics.csv",other)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.2))
    labels={"c1":"Reizman Case1","c2":"Reizman Case2","c3":"Reizman Case3","c4":"Reizman Case4","hplc":"HPLC"}
    for ax,row in zip(axes.flat,side):
        task=row["task"]
        entries=[("Current (0+20)",row["current_SearchAUC_mean"],row["current_SearchAUC_SD"],"#AA3377")]
        for name,label,color in (("published_LLAMBO","Published LLAMBO*","#66CCEE"),("published_FGSS","Published FGSS","#4477AA"),("published_GP_EI","Published GP-EI","#228833")):
            mean=row[f"{name}_SearchAUC_mean"]
            if mean!="":entries.append((label,mean,row[f"{name}_SearchAUC_SD"],color))
        for pos,(label,mean,sd,color) in enumerate(entries):
            ax.errorbar(mean,pos,xerr=sd,fmt="o",capsize=3,color=color)
        ax.set_yticks(range(len(entries)),[e[0] for e in entries],fontsize=8)
        ax.invert_yaxis();ax.set_title(labels[task]);ax.set_xlabel("Search AUC (task units)");ax.grid(axis="x",alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.9,"Different protocols",weight="bold",fontsize=13)
    axes.flat[-1].text(0,.78,"Current: fixed pool,0+20,4 runs.\nPublished Reizman: continuous search,\n8+20,6 runs. HPLC setup not fully\nspecified in the cited appendix.\n\nDots: means. Error bars: sample SD.\n* Adapted LLAMBO in FGSS paper.\n\nNo controlled ranking is inferred.",va="top",fontsize=10,linespacing=1.5)
    fig.suptitle("Published metrics alongside current results",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ("png","pdf"):fig.savefig(OUT/f"published_side_by_side.{suffix}",dpi=160)
    plt.close(fig)
    manifest={str(p.relative_to(OUT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="sha256.json"}
    (OUT/"sha256.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(published_search_metric_rows=len(metrics),other_metric_rows=len(other),
                         papers=len(sources),new_experiments=0,new_LLM_calls=0,output=str(OUT))))


if __name__=="__main__":main()
