"""Write the final reading guide only after numerical and visual review."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13_late'


def main():
    read=lambda name:json.loads((P/name).read_text(encoding='utf-8'))
    a=read('audit/new_native_replay.json');v=read('audit/visual_review.json');c=read('data/conditional_facts.json');h=read('data/holdout_facts.json')
    if a['status']!='PASS_COMPLETE' or not v['all_pages_visually_reviewed'] or v['unresolved_issues']:raise ValueError('Numerical/visual completion required')
    counts=lambda x:f"{x['wins']} wins, {x['ties']} ties, {x['losses']} losses"
    zhcounts=lambda x:f"{x['wins']}胜、{x['ties']}平、{x['losses']}负"
    text='''Frozen V13: additive late-stage validation manuscript

Read main.pdf first; supplement.pdf retains all historical starts, both new LLM
realizations per continuation arm, prior V12 removal experiments and execution
limits. This is a complete registered computational study and research draft,
not a claim of universal superiority, wet-laboratory validation or journal acceptance.
The original paper/v13 release and all previous 4+16 results remain unchanged.

Evidence scope
Original: 227 method trajectories, including 64 V13 trajectories across eight
tasks and two distinct budgets. Warm uses 4 shared observations +16 queries;
cold uses zero supplied observations +20 paid queries. Backend-mixed warm
development evidence is labeled separately. The old development catalogue is
retained at its original cutoff, not counted as independent confirmation.
Additional: 66 ten-query continuations from six exact query-ten prefixes, plus
9 Direct Arylation trajectories from zero observations. 75 trajectories and
840 new paid outcomes all pass native replay. The full twenty-query cost
includes each inherited prefix. BO has one deterministic tail; each LLM arm
has two fresh response realizations per prefix. No outcome-selected reruns.

Important files
main.tex, late_methods.tex, supplement.tex, late_supplement.tex: editable LaTeX.
data/metrics.csv and rounds.csv: unchanged historical publication arithmetic.
data/final_metrics.csv and final_rounds.csv: every new trajectory and paid query.
data/conditional_effects.csv: all 30 full-minus-control mean contrasts.
data/late_forecast_calculations.csv: scalar predictive updates and permissions.
data/mechanism_traces.json: public rationales and decision-source paths.
data/candidate_coverage_*.csv: post-hoc scope opportunity, not causal effects.
data/new_fit_inventory.csv: all 840 new prefixes, 831 fitted and 9 no-data.
data/LLM_request_inventory.csv: actual fresh-agent assignment and raw-byte custody.
data/new_costs.json: call counts; token and monetary costs are not exposed.
audit/: native replay, forecast arithmetic, PDF rendering and visual-review receipts.

Portable arithmetic (standard library only; no model or API)
  python verify_results.py
  python verify_results.py --workspace C:/path/to/Chemsys
The second form additionally verifies original state, commitment, raw-response
and numerical-checkpoint hashes in the local evidence tree. Neither command
fits a GP, contacts an LLM or selects an experiment.

Figure/table regeneration, in a working COPY of the source package
  python regenerate_historical_assets.py
  python regenerate_new_assets.py --fresh
Requires NumPy and Matplotlib. Rebuilding figures changes their PDF metadata
and therefore invalidates the old artifact manifest; retain the sealed release.
Compile main.tex and supplement.tex with Tectonic from their own directory.

Project audit/build entrypoints (from the Chemsys root)
  code/analysis/audit_v13_late_validation.py
  code/publication_v13_late/assets.py --fresh
  code/publication_v13_late/traces.py --fresh
  code/publication_v13_late/coverage.py --fresh
  code/publication_v13_late/bind.py
  code/publication_v13_late/render.py
  code/publication_v13_late/typeset.py
Each scientific source is guarded by its registration; publication helpers do
not alter the frozen controller. Do not relaunch the completed experimental
batch to reproduce arithmetic. Native evidence is separately sealed under
experiments/milestones/v13_late_validation_20261103 and linked to the original
V13 milestone. Credentials, environment binaries and hosted model weights
are excluded from the paper package.

Boundaries
Requested inference is gpt-6.1-terra medium, fresh agent per request/repair;
serving weights and intrinsic random seeds are not independently attested.
The independent task is held out from development, not proven absent from
pretraining. Same public packets can have different opaque routing metadata.
Module contrasts retain fresh-response variation and selected-context bias.
The local Reasoning BO control adapts the official selection component; it
does not reproduce the original initialization, batch and external-store setup.
Author identities, affiliations, funding, competing interests and submission
forms must be supplied by the investigators before journal submission.
'''
    (P/'README.txt').write_text(text+'\nConditional full V13 versus BO: '+counts(c['bo']['delta_continuation_SearchAUC'])+' in tail SearchAUC.\nConditional full V13 versus Reasoning BO: '+counts(c['rbo']['delta_continuation_SearchAUC'])+'.\nDirect Arylation versus BO: '+counts(h['bo']['SearchAUC20'])+' in SearchAUC.\n',encoding='utf-8')
    zh='''本次交付保留冻结V13，原4＋16与0＋20结果及旧版论文均未覆盖。

阅读顺序：main.pdf（英文主稿）→ supplement.pdf（完整结果与机制边界）→ data/（全精度CSV和逐轮证据）。
新增实验为6个相同10点前缀上的66条续跑，加独立Direct Arylation任务9条轨迹，共75条、840次新查询，均完成审计。
SearchAUC指标保持；同时报告最终最佳、后半程增量及回顾性95%目标达成率。相同前缀下的尾段AUC差正好是完整20轮AUC差的两倍，并非另一个独立证据。
KF冻结、停止刷新、重新耦合生命周期均有明确定义。它们保留两次独立LLM响应，不选择最好的一次，也不能将差值全部解释为单模块因果效应。
历史Perera和HPLC有后期收益；Doyle的重要优势更早形成。独立Direct Arylation的V13最佳值均在前8次实验已达到，不能用它支持后期增益。
原始失败、知识覆盖不足、GP本身找到好点和暂停知识仅作警示的案例均保留。
这是证据完整的计算研究稿；不等于已证明稳定超越所有对照，也不包含新的真实实验验证。

复核：在本目录运行 python verify_results.py，不需要模型、API或重新做优化。
'''
    zh+='\n本次相同前缀下，完整V13的尾段SearchAUC相对BO为'+zhcounts(c['bo']['delta_continuation_SearchAUC'])+'；相对Reasoning BO为'+zhcounts(c['rbo']['delta_continuation_SearchAUC'])+'。\n'
    zh+='相对固定KF为'+zhcounts(c['kf_fixed']['delta_continuation_SearchAUC'])+'；相对停止刷新为'+zhcounts(c['no_refresh']['delta_continuation_SearchAUC'])+'。本轮没有证明这两个模块带来稳定的SearchAUC增量。\n'
    zh+='Direct Arylation相对BO的SearchAUC为'+zhcounts(h['bo']['SearchAUC20'])+'，相对Reasoning BO为'+zhcounts(h['rbo']['SearchAUC20'])+'。\n'
    (P/'中文导读.txt').write_text(zh,encoding='utf-8');print('Final reading guides written.')


if __name__=='__main__':main()
