"""Create an index of included COMPLETE trajectories; retain cohort boundaries."""
import csv
import json
from pathlib import Path
import statistics as st
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    if not (ROOT/'experiments/runs/hypothesis_scope_v2_20261105').exists():
        raise SystemExit('Exporting the raw-state index requires local records. See LOCAL_DATA_REQUIREMENTS.md.')
    with (ROOT/'data/tasks.csv').open(encoding='utf-8',newline='') as f:catalogue={r['task']:r for r in csv.DictReader(f)}
    aliases={s:s for s in catalogue}
    aliases.update({f'Reizman_Suzuki_Case_{i}':f'c{i}' for i in range(1,5)})
    aliases.update({'Perera_Suzuki':'perera','Doyle_Buchwald_Hartwig':'doyle','SnAr':'snar','HPLC':'hplc','Direct_Arylation':'aryl'})
    for s,r in catalogue.items():
        p=json.loads((ROOT/r['pool']).read_text(encoding='utf-8'))
        if 'task_id' in p:aliases[p['task_id']]=s
    output=[];excluded=[]
    def add(state,cohort,location,guess=None):
        history=state.get('history',[])
        if state.get('status')!='COMPLETE' or len(history)!=20:
            excluded.append(dict(location=location,status=state.get('status'),observations=len(history)));return
        short=aliases.get(state.get('task'),guess)
        assert short in catalogue,(location,state.get('task'))
        ids=[h['candidate_id'] for h in history];assert len(set(ids))==20,location
        values=[float(h['raw_value']) for h in history];best=[max(values[:i+1]) for i in range(20)]
        top=float(catalogue[short]['maximum_raw'])
        output.append(dict(cohort=cohort,task=short,method=state.get('arm',state.get('method','stored')),stream=state.get('seed',state.get('stream','')),
                           observations=20,SearchAUC_raw=st.mean(best),SearchAUC_pct_of_max=100*st.mean(best)/top,
                           final_best_raw=best[-1],final_best_pct_of_max=100*best[-1]/top,
                           first_candidate=ids[0],first_value=values[0],location=location))
    base=ROOT/'experiments/runs/hypothesis_scope_v2_20261105'
    for p in sorted(base.rglob('state.json')):
        rel=p.relative_to(base);parts=rel.parts
        guess=next((x for x in parts if x in catalogue),None)
        add(json.loads(p.read_text(encoding='utf-8-sig')),'primary/'+parts[0],p.relative_to(ROOT).as_posix(),guess)
    for p in sorted((ROOT/'evidence/deepseek/new').rglob('state.json')):
        parts=p.relative_to(ROOT/'evidence/deepseek/new').parts
        add(json.loads(p.read_text(encoding='utf-8-sig')),'deepseek/'+parts[0],p.relative_to(ROOT).as_posix(),next((x for x in parts if x in catalogue),None))
    for p in sorted((ROOT/'evidence/deepseek/local_bo').rglob('state.json')):
        parts=p.relative_to(ROOT/'evidence/deepseek/local_bo').parts
        add(json.loads(p.read_text(encoding='utf-8-sig')),'deepseek/local_bo',p.relative_to(ROOT).as_posix(),parts[0])
    for p in sorted((ROOT/'archives').glob('run__hypothesis_scope*.zip')):
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if name.endswith('/state.json'):
                    guess=name.split('/')[-2].split('_')[0]
                    add(json.loads(z.read(name).decode('utf-8-sig')),'development/controller',p.relative_to(ROOT).as_posix()+'::'+name,guess)
    write(ROOT/'results/trajectory_metrics.csv',output)
    (ROOT/'results/trajectory_metrics_notes.json').write_text(json.dumps(dict(complete_included_states=len(output),excluded_incomplete_states=excluded,N=20,formula='mean(cumulative_maximum(raw_values)); normalized by the full-pool maximum only in evaluation',cohorts_are_not_pooled=True),indent=2)+'\n',encoding='utf-8')
    print('Indexed',len(output),'complete trajectories;',len(excluded),'incomplete states excluded.')

if __name__=='__main__':main()
