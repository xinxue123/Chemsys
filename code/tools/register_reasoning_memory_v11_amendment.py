"""Register a non-selection citation parser fix, retaining the initial freeze."""
from datetime import datetime,timezone
from pathlib import Path
import json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from reasoning_memory_v11 import study
from chem.benchmarks.knowledge_llm import digest


def main():
    base=study.directory(ROOT);target=base/'technical_amendment_01.json'
    if target.exists():raise RuntimeError('Amendment already registered; cannot overwrite')
    original=study.old.read(base/'protocol.json');current=study.sources(ROOT)
    changed=[name for name,sha in current.items() if original['source_sha256'].get(name)!=sha]
    allowed=['code/experiment_tools/reasoning_memory_v11/llm.py','code/experiment_tools/reasoning_memory_v11/study.py',
        'code/experiment_tools/reasoning_memory_v11/tests/test_pipeline.py']
    if set(changed)!=set(allowed):raise ValueError('Unexpected source changes in citation-only amendment')
    snapshot=ROOT/'runtime/knowledge_loop/snapshots/v11_reasoning_memory_amendment01'
    paths=[p for k in ('runs','requests') for p in (base/k).rglob('*') if p.is_file() and p.suffix!='.lock']
    custody={};states={}
    for source in paths:
        name=source.relative_to(base).as_posix();dest=snapshot/'pre_amendment_evidence'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);custody[name]=study.old.sha(dest)
    for source in (base/'runs').glob('*/state.json'):
        s=study.old.read(source);states[s['task']]=dict(completed_queries=len(s['decisions']),status=s['status'],state_sha256=study.old.sha(source))
    for name in changed:
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    a=dict(kind='NON_SELECTION_CITATION_TRANSPORT_FIX',created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=original['protocol_sha256'],original_core_commit=original['code_milestone'],
        revised_core_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        changed_source_files=changed,revised_source_sha256=current,pre_amendment_states=states,
        pre_amendment_artifacts_sha256=custody,snapshot=snapshot.relative_to(ROOT).as_posix(),
        reason='LLM cited Notes already supplied in forecast_notes but outside retrieved scope. The citation-only sidecar validator rejected these without querying outcomes.',
        fix='Accept exact Note ID/version present in retrieved_notes OR nonretracted forecast_notes; retain raw citations and separately report retrieval membership.',
        unchanged=['official Insight schema and confidence extractor','system prompts and candidate menus','GP and acquisition','KF equations and note semantics','event schedule','all completed decisions'],
        recovery='Use earliest previously saved response that satisfies amended non-selection annotation validator; keep failed repairs. Never rerun a completed query.',
        performance_tuning=False,new_oracle_queries_during_registration=0,new_API_calls_during_registration=0)
    a['amendment_sha256']=digest(a);study.write(target,a);study.write(snapshot/'technical_amendment_01.json',a)
    study.check(ROOT)
    print(json.dumps(dict(amendment_sha256=a['amendment_sha256'],states=states),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
