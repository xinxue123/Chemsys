"""Input-only molecular features, once per catalog; never load yield oracles."""
import json
from pathlib import Path
import sys
import numpy as np
from . import reasoning_bo_data as data
from .suzuki_domain import digest

ROOT=Path('data/benchmarks/chemical_features_20261029')
PARAMETERS={'rdkit':'2024.09.6','fingerprint':'Morgan','radius':2,'bits':2048,'includeChirality':True,
    'salts':'all supplied fragments retained; no salt removal or charge neutralization'}


def prepare():
    sys.path.insert(0,str(Path('outputs/rdkit_runtime').resolve()))
    import rdkit
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    assert rdkit.__version__==PARAMETERS['rdkit']
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    files=[];counts={}
    for task in data.SPECS:
        pool=data.load_pool(task);axes={}
        for axis in pool['axes']:
            records=[];fps=[]
            for value in pool['levels'][axis]:
                molecule=Chem.MolFromSmiles(value)
                if molecule is None:raise ValueError(f'invalid supplied molecular graph: {task}/{axis}')
                fp=generator.GetFingerprint(molecule);fps.append(fp)
                records.append({'input':value,'canonical_SMILES':Chem.MolToSmiles(molecule),
                    'on_bits':list(fp.GetOnBits()),'atom_count':molecule.GetNumAtoms(),
                    'elements':{s:sum(a.GetSymbol()==s for a in molecule.GetAtoms()) for s in sorted({a.GetSymbol() for a in molecule.GetAtoms()})}})
            similarities=np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps])
            assert np.array_equal(similarities,similarities.T) and np.all(similarities.diagonal()==1.)
            axes[axis]={'levels':records,'Tanimoto':similarities.tolist(),
                'unique_canonical_structures':len({r['canonical_SMILES'] for r in records}),
                'unique_fingerprints':len({tuple(r['on_bits']) for r in records}),
                'smallest_eigenvalue':float(np.linalg.eigvalsh(similarities).min())}
        payload={'task':task,'parameters':PARAMETERS,'pool_sha256':digest(pool),'generator_sha256':data.filehash(__file__),
            'source':'public measured-candidate structures only; no yield observations or oracle files used','axes':axes}
        payload['feature_sha256']=digest(payload);path=ROOT/(task+'.json')
        if path.exists():assert data.read(path)==payload
        else:data.write(path,payload)
        files.append({'path':path.as_posix(),'sha256':data.filehash(path)})
        counts[task]={a:len(r['levels']) for a,r in axes.items()}
    manifest={'status':'INPUT_FEATURES_COMPLETE','parameters':PARAMETERS,'files':files,'counts':counts,
        'source_URL':'https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html',
        'operator_compiles_in_loop':0,'GP_environment_dependencies_unchanged':True,'no_oracle_read':True}
    data.write(ROOT/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':prepare()
