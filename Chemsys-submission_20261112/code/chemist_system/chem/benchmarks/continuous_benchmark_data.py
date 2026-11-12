"""Input-only finite pools and isolated outcome providers for SnAr / Olympus HPLC."""
from pathlib import Path
from hashlib import sha256
import csv
import io
import itertools
import json
import numpy as np

ROOT=Path('data/benchmarks/continuous_20261029')
OUT=Path('research/continuous_20261029')
OLYMPUS='440b6b58ebfcaa2391cff7e94b570fb4fda98d68'
SNAR_AXES=['tau','equiv_pldn','conc_dfnb','temperature']
HPLC_AXES=['sample_loop','additional_volume','tubing_volume','sample_flow','push_speed','wait_time']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()

def fetch():
    from urllib.request import urlopen,Request
    source=ROOT/'sources';source.mkdir(parents=True,exist_ok=True)
    base=f'https://raw.githubusercontent.com/aspuru-guzik-group/olympus/{OLYMPUS}/'
    files={'hplc.csv':'src/olympus/datasets/dataset_hplc/data.csv',
           'hplc_description.txt':'src/olympus/datasets/dataset_hplc/description.txt',
           'hplc_config.json':'src/olympus/datasets/dataset_hplc/config.json','olympus.LICENSE':'LICENSE'}
    receipt=[]
    for name,rel in files.items():
        p=source/name;url=base+rel
        if not p.exists():p.write_bytes(urlopen(Request(url,headers={'User-Agent':'ChemSys-research'}),timeout=40).read())
        receipt.append({'path':p.as_posix(),'url':url,'sha256':filehash(p),'bytes':p.stat().st_size})
    write(OUT/'source_receipt.json',{'olympus_commit':OLYMPUS,'files':receipt})
    return receipt

def _pool(task,axes,X,lo,hi,scale,unit,description,extra=None):
    X=np.asarray(X,dtype=float);lo=np.array(lo);hi=np.array(hi)
    assert np.isfinite(X).all() and (hi>lo).all()
    z=(X-lo)/(hi-lo);assert (z>=0).all() and (z<=1).all()
    # IDs are names only; independent priorities resolve all exact score ties.
    ids=[f'{task}_{i:05d}' for i in range(len(X))]
    initial=np.random.default_rng(0).choice(len(X),4,replace=False).tolist()
    pool={'task':task,'axes':axes,'bounds':{a:[float(l),float(h)] for a,l,h in zip(axes,lo,hi)},
          'z':z.tolist(),'points':[{a:float(v) for a,v in zip(axes,row)} for row in X],
          'ids':ids,'initial_ids':[ids[j] for j in initial], 'objective_scale':scale,'objective_unit':unit,
          'description':description,'all_continuous':True,**(extra or {})}
    write(ROOT/task/'pool.json',pool);return pool

def prepare():
    lo=[.5,1.,.1,30.];hi=[2.,5.,.5,120.]
    grid=np.array(list(itertools.product(*[np.linspace(l,h,8) for l,h in zip(lo,hi)])))
    snar=_pool('SnAr',SNAR_AXES,grid,lo,hi,13000.,'kg m^-3 h^-1',
        'Plug-flow SnAr: 2,4-dinitrofluorobenzene + pyrrolidine; maximize space-time yield (STY). Residence time in min, equivalents dimensionless, inlet concentration in M, temperature in Celsius. E-factor is recorded but not optimized.',
        {'objective_name':'sty','secondary_objective':'e_factor','oracle_noise_percent':0.,'candidate_recipe':'8 equally spaced physical levels per axis, Cartesian product; no outcome filtering'})
    raw=np.loadtxt(ROOT/'sources/hplc.csv',delimiter=',');assert raw.shape==(1386,7) and np.isfinite(raw).all()
    # Same exact-match semantics as the official LGBO HPLC evaluator: first row.
    # Preserve every source measurement separately; do not turn replicates into choices.
    _,first,counts=np.unique(raw[:,:6],axis=0,return_index=True,return_counts=True)
    first=np.sort(first);X=raw[first,:6]
    config=read(ROOT/'sources/hplc_config.json')
    hplc=_pool('HPLC',HPLC_AXES,X,raw[:,:6].min(0),raw[:,:6].max(0),10000.,'peak area (source arbitrary units)',
        'HPLC sample-injection protocol: maximize measured chromatographic peak area by adjusting sample loop, additional volume, tubing volume, sample flow, push speed and wait time. The target is detector peak area, not reaction yield.',
        {'objective_name':'peak_area','source_row_indices_0based':first.tolist(),
         'candidate_recipe':'unique measured six-input vectors, first source row outcome at exact match; no off-table interpolation',
         'normalization_note':'fixed engineering scale 10000 peak-area units; no full-target min/max used in decisions'})
    p=ROOT/'HPLC/oracle.npz';np.savez_compressed(p,values=raw[first,6]/hplc['objective_scale'],raw_values=raw[first,6])
    duplicates=[]
    for row in X:
        ix=np.flatnonzero(np.all(raw[:,:6]==row,axis=1))
        if len(ix)>1:duplicates.append({'rows_0based':ix.tolist(),'objective_values':raw[ix,6].tolist()})
    quality={'SnAr':{'candidates':len(grid),'axes':4,'objective':'STY; E-factor auxiliary','noise_percent':0},
        'HPLC':{'source_rows':len(raw),'source_columns':7,'unique_inputs':len(X),'duplicate_extra_rows':len(raw)-len(X),
                'duplicate_input_groups':len(duplicates),'conflicting_replicate_groups':sum(len(set(r['objective_values']))>1 for r in duplicates),
                'missing_values':int(np.isnan(raw).sum()),'finite':bool(np.isfinite(raw).all()),'raw_target_range':[float(raw[:,6].min()),float(raw[:,6].max())],
                'duplicates':duplicates,'handling':'first source row; all raw records retained; no best-replicate selection','upstream_config':config},
        'leakage_policy':'pool and initialization depend only on inputs; no unseen outcomes or full-pool target statistics in LLM/GP decisions; raw ranges here are retrospective quality metadata only'}
    write(OUT/'data_quality.json',quality)
    return {'SnAr':len(snar['ids']),'HPLC':len(hplc['ids']),'HPLC_duplicate_extra_rows':len(raw)-len(X)}

def snar_oracle():
    # Importing Summit eagerly requires unrelated optional optimizer packages.
    # Execute the UNCHANGED official class AST instead; only framework setup is
    # bypassed. These two official numerical methods need numpy and solve_ivp.
    import ast
    from scipy.integrate import solve_ivp
    source=Path('.venv-summit/Lib/site-packages/summit/benchmarks/snar.py')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SnarBenchmark')
    namespace={'np':np,'solve_ivp':solve_ivp,'Experiment':object}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    cls=namespace['SnarBenchmark'];b=cls.__new__(cls);b.noise_level=0.;b.rng=np.random.default_rng(0)
    pool=read(ROOT/'SnAr/pool.json')
    result=[b._integrate_equations(*(row[a] for a in SNAR_AXES))[:2] for row in pool['points']]
    a=np.array(result);assert np.isfinite(a).all()
    np.savez_compressed(ROOT/'SnAr/oracle.npz',values=a[:,0]/pool['objective_scale'],raw_values=a[:,0],e_factor=a[:,1])
    write(OUT/'snar_oracle_receipt.json',{'source_path':str(source),'sha256':filehash(source),'points':len(a),
         'call':'unchanged Summit SnarBenchmark class AST; _integrate_equations and _integrand; framework __init__ bypassed, rng and noise_level initialized explicitly','noise_level':0.,
         'objective':'sty','secondary_recorded':'e_factor','values_sha256':filehash(ROOT/'SnAr/oracle.npz')})
    return {'status':'PASS','points':len(a)}

def reveal(task,key):
    pool=read(ROOT/task/'pool.json');j=pool['ids'].index(key)
    with np.load(ROOT/task/'oracle.npz') as f:
        out={'candidate_id':key,'y':float(f['values'][j]),'raw_value':float(f['raw_values'][j])}
        if 'e_factor' in f:out['e_factor']=float(f['e_factor'][j])
    return out

if __name__=='__main__':
    import sys
    print(json.dumps({'fetch':fetch,'prepare':prepare,'snar_oracle':snar_oracle}[sys.argv[1]](),ensure_ascii=False),flush=True)
