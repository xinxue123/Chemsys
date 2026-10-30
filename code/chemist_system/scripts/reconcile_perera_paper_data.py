"""Evaluation-only original-to-author-table reconciliation; no policy imports."""
from collections import Counter
from pathlib import Path
import csv
import io
import json
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks.hte_workbook import table
from chem.benchmarks.reaction_transfer_data import write


def main():
    raw=table('data/benchmarks/reaction_transfer/raw/Perera_Suzuki.xlsx','Sheet1')
    with zipfile.ZipFile('third_party/reasoning_bo_official/data/chemistry_data.zip') as z:
        official=list(csv.DictReader(io.StringIO(z.read('suzuki.csv').decode('utf-8-sig'))))
    axes=['Electrophile_SMILES','Nucleophile_SMILES','Ligand_SMILES','Base_SMILES','Solvent_SMILES']
    choices={a:sorted({r[a] for r in official}) for a in axes}
    def only(axis,predicate):
        rows=[s for s in choices[axis] if predicate(s)]
        assert len(rows)==1,(axis,rows)
        return rows[0]
    electrophile={name:only(axes[0],lambda s,p=prefix:s.startswith(p)) for name,prefix in
        [('6-Bromoquinoline','Br'),('6-chloroquinoline','Cl'),('6-Iodoquinoline','I'),('6-triflatequinoline','O=S')]}
    nucleophile={name:only(axes[1],lambda s,k=marker:k in s) for name,marker in
        [('2a, Boronic Acid','B(O)O'),('2b, Boronic Ester','B4OC'),('2c, Trifluoroborate','[B-]')]}
    ligand={name:only(axes[2],lambda s,p=prefix:s.startswith(p)) for name,prefix in [
        ('AmPhos','CC(C)(C)P'),('CataCXium A','CCCCP'),('P(Cy)3','P(C1CCCCC1)'),
        ('P(Ph)3','P(C1=CC=CC=C1)'),('P(o-Tol)3','CC1=CC=CC=C1P'),
        ('P(tBu)3','CC(P('),('SPhos','COC1='),('XPhos','CC(C1='),('Xantphos','CC1(C)C2='),
        ('dppf','[c-]'),('dtbpf','CC(C)(P(')]}
    base={'CsF':'[Cs+].[F-]','Et3N':'CCN(CC)CC','K3PO4':'O=P([O-])([O-])[O-].[K+].[K+].[K+]',
        'KOH':'[K+].[OH-]','NaOH':'[Na+].[OH-]','NaHCO3':'OC([O-])=O.[Na+]',
        'LiOtBu':'CC([O-])C.[Li+]'}
    solvent={'DMF':'O=CN(C)C','MeCN':'N#CC','MeOH':'CO','THF':'C1COCC1'}
    maps=[electrophile,nucleophile,ligand,base,solvent]
    original_columns=['Reactant_1_Name','Reactant_2_Name','Ligand_Short_Hand','Reagent_1_Short_Hand','Solvent_1_Short_Hand']
    selected=[];excluded=Counter()
    for r in raw:
        names=[str(r[k]).strip() for k in original_columns]
        if any(v not in m for v,m in zip(names,maps)):
            excluded['reverse_polarity_or_blank_control']+=1;continue
        selected.append((tuple(m[v] for v,m in zip(names,maps)),float(r['Product_Yield_PCT_Area_UV'])))
    a={key:y for key,y in selected};b={tuple(r[k] for k in axes):float(r['yield']) for r in official}
    common=set(a)&set(b)
    errors=[abs(a[k]-b[k]) for k in common]
    rounded_matches=sum(round(a[k],2)==b[k] for k in common)
    result={'status':'PASS_AUTHOR_ROUNDING_2DP' if len(common)==3696 and set(a)==set(b) and rounded_matches==3696 else 'DIFFERENCES',
        'original_rows':len(raw),'selected_original_rows':len(selected),'official_rows':len(official),
        'matched_condition_tuples':len(common),'excluded_original_rows':dict(excluded),
        'maximum_absolute_yield_difference_percent':max(errors),
        'all_author_yields_equal_original_rounded_to_2dp':rounded_matches==3696,
        'rounded_matches':rounded_matches,'oracle_policy':'Use author CSV two-decimal yields exactly; original higher-precision UV-area values retained separately.',
        'axis_label_maps':dict(zip(axes,maps)),
        'source_annotation_limit':[{'raw_label':'LiOtBu','author_SMILES':'CC([O-])C.[Li+]',
            'observation':'Author structure string has a three-carbon alkoxide fragment whereas the raw label says tBu. Kept author representation for the registered paper comparison; not silently corrected using outcomes.',
            'scope':'All methods share this domain. Chemical interpretation of this categorical option requires resolving the source annotation separately.'}],
        'policy_use':'None: evaluation-only reconciliation; maps and unseen outcomes are never sent to a reasoner.'}
    write('research/reasoning_bo_comparison/perera_original_reconciliation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('axis_label_maps','source_annotation_limit')},indent=2))


if __name__=='__main__':main()
