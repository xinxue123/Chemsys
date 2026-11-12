"""Curated experimental definitions, with no benchmark result text in packets."""
from copy import deepcopy
from pathlib import Path
import sys
from chem.benchmarks.knowledge_domain_context import task_context

ROOT=Path(__file__).resolve().parents[3]
SOURCE='experiments/metadata/chemical_background_20261105/sources'
SOURCES=dict(
    reizman=dict(url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5123644/fullTextXML',
        doi='10.1039/C6RE00153J',allowed_sections=['Scheme1','substrate identity images u1-u8'],
        excluded_sections=['all outcome tables','optimization results','response surfaces','kinetic fits','conclusions']),
    doyle=dict(url='https://pubmed.ncbi.nlm.nih.gov/29449509/',doi='10.1126/science.aar5169',
        allowed_sections=['abstract: reaction identity and fixed amine']),
    snar=dict(local='runtime/environments/.venv-summit/Lib/site-packages/summit/benchmarks/snar.py',
        allowed_sections=['class reaction description','input domain only'],excluded_sections=['kinetic constants','objective function implementation']),
    hplc=dict(local='experiments/datasets/continuous_20261029/sources/hplc_description.txt',
        allowed_sections=['input meanings and units','introductory process definition']))

LIGANDS={
    'L1':dict(name='XPhos',class_description='dicyclohexyl-substituted dialkylbiaryl monophosphine',phosphorus_donors=1),
    'L2':dict(name='SPhos',class_description='dicyclohexyl-substituted dimethoxybiaryl monophosphine',phosphorus_donors=1),
    'L3':dict(name='RuPhos',class_description='dicyclohexyl-substituted diisopropoxybiaryl monophosphine',phosphorus_donors=1),
    'L4':dict(name='Xantphos',class_description='xanthene-based bidentate bis(diphenylphosphine)',phosphorus_donors=2),
    'L5':dict(name='tricyclohexylphosphine (PCy3)',class_description='trialkyl monophosphine',phosphorus_donors=1),
    'L6':dict(name='triphenylphosphine (PPh3)',class_description='triaryl monophosphine',phosphorus_donors=1),
    'L7':dict(name='tri-tert-butylphosphine (PtBu3)',class_description='trialkyl monophosphine',phosphorus_donors=1)}
SUBSTRATES={
    1:[dict(role='electrophile',name='3-bromoquinoline',SMILES='Brc1cnc2ccccc2c1',source_image='c6re00153j-u1.jpg'),
       dict(role='organoboron_partner',name='3,5-dimethylisoxazole-4-boronic acid pinacol ester',SMILES='Cc1noc(C)c1B1OC(C)(C)C(C)(C)O1',source_image='c6re00153j-u2.jpg')],
    2:[dict(role='electrophile',name='3-chloropyridine',SMILES='Clc1cccnc1',source_image='c6re00153j-u3.jpg'),
       dict(role='organoboron_partner',name='3,5-dimethylisoxazole-4-boronic acid pinacol ester',SMILES='Cc1noc(C)c1B1OC(C)(C)C(C)(C)O1',source_image='c6re00153j-u4.jpg')],
    3:[dict(role='electrophile',name='3-chloropyridine',SMILES='Clc1cccnc1',source_image='c6re00153j-u5.jpg'),
       dict(role='organoboron_partner',name='benzofuran-2-boronic acid',SMILES='OB(O)c1cc2ccccc2o1',source_image='c6re00153j-u6.jpg')],
    4:[dict(role='electrophile',name='2-chloropyridine',SMILES='Clc1ccccn1',source_image='c6re00153j-u7.jpg'),
       dict(role='organoboron_partner',name='N-Boc-pyrrole-2-boronic acid',SMILES='CC(C)(C)OC(=O)n1cccc1B(O)O',source_image='c6re00153j-u8.jpg')]}
COMMON_NAMES={
    'C1COCC1':'tetrahydrofuran (THF)','CO':'methanol','N#CC':'acetonitrile','O=CN(C)C':'N,N-dimethylformamide (DMF)',
    '[Cs+].[F-]':'cesium fluoride','[K+].[OH-]':'potassium hydroxide','[Na+].[OH-]':'sodium hydroxide',
    'CC([O-])C.[Li+]':'lithium isopropoxide','CCN(CC)CC':'triethylamine',
    'O=P([O-])([O-])[O-].[K+].[K+].[K+]':'potassium phosphate',
    'OC([O-])=O.[Na+]':'sodium hydrogen carbonate',
    'P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3':'triphenylphosphine (PPh3)',
    'P(C1CCCCC1)(C2CCCCC2)C3CCCCC3':'tricyclohexylphosphine (PCy3)',
    'CC(P(C(C)(C)C)C(C)(C)C)(C)C':'tri-tert-butylphosphine (PtBu3)',
    'CN1CCCN2C1=NCCC2':'1,8-diazabicyclo[5.4.0]undec-7-ene (DBU)'}


def molecular_graph(smiles):
    sys.path.insert(0,str(ROOT/'runtime/tools/rdkit_runtime')) if str(ROOT/'runtime/tools/rdkit_runtime') not in sys.path else None
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid supplied molecular structure')
    counts={symbol:sum(a.GetSymbol()==symbol for a in mol.GetAtoms()) for symbol in ('P','B','N','O','F','Cl','Br','I','Fe')}
    return dict(canonical_SMILES=Chem.MolToSmiles(mol),formula=rdMolDescriptors.CalcMolFormula(mol),
        atom_counts=counts,formal_charge=Chem.GetFormalCharge(mol),
        aromatic_atom_count=sum(a.GetIsAromatic() for a in mol.GetAtoms()),
        annotation_type='deterministic molecular graph facts, not inferred activity or yield')


def context(pool):
    public=task_context(pool.task,pool);public.pop('source_references',None)
    # Hide the named benchmark in decision-agent transport. The chemistry itself
    # is provided in full, including fixed substrates and molecular identities.
    public.pop('task',None)
    public['identity_evidence']='Curated experimental definitions and supplied input structures; no benchmark performance conclusions.'
    if pool.group=='P0':
        substrates=deepcopy(SUBSTRATES[int(pool.task[-1])])
        for row in substrates:
            row.pop('source_image',None);row['structure_status']='manually transcribed from published substrate identity drawing'
            row['graph']=molecular_graph(row['SMILES'])
        public['fixed_reactants']=substrates
        public['substrate_scope']['identity_status']='fixed reactant names and structures supplied'
        public['substrate_scope']['statement']='The two listed reactants are fixed; do not change substrate identity.'
        public['fixed_conditions']=dict(base='DBU',base_equivalents=2.,organoboron_equivalents=1.5,
            medium='THF/water 5:1 by volume',precatalyst_scaffold='2-aminobiphenyl palladacycle family',
            note='These are source experimental definitions; only the four registered variables may change.')
        item=next(v for v in public['inputs'] if v['name']=='catalyst')
        item.update(representation='verified precatalyst scaffold and ligand combination',identity_status='decoded from source structure scheme')
        item.pop('interpretation_limit',None)
        item['chemical_identities']={}
        for value in item['levels']:
            scaffold,ligand=value.split('-')
            item['chemical_identities'][value]=dict(scaffold=scaffold,ancillary_group='methanesulfonate (OMs)' if scaffold=='P1' else 'chloride (Cl)',
                ligand=deepcopy(LIGANDS[ligand]),meaning='chemical combination; labels carry no ordinal relation')
    elif pool.group=='P1':
        for variable in public['inputs']:
            variable['chemical_identities']={value:dict(name=COMMON_NAMES.get(value,'name not assigned; exact molecular graph supplied'),
                SMILES=value,graph=molecular_graph(value)) for value in variable['levels']}
        if pool.task=='Doyle_Buchwald_Hartwig':
            public['fixed_reactants']=[dict(role='amine',name='4-methylaniline (p-toluidine)',SMILES='Cc1ccc(N)cc1')]
            public['substrate_scope']['identity_status']='variable aryl-halide structures and fixed 4-methylaniline supplied'
            public['substrate_scope']['statement']='Aryl halide varies; 4-methylaniline is fixed. Do not introduce another amine.'
            public['experimental_design']='Additives are intentionally varied potentially inhibitory heterocycles, not additional coupling partners.'
        public['unknown_fixed_details']=['temperature, time, precursor loading and ancillary fixed protocol details are not specified by this input table; do not vary or invent them']
    elif pool.task=='SnAr':
        public['fixed_reactants']=[dict(role='electrophile',name='2,4-dinitrofluorobenzene',SMILES='Fc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]'),
            dict(role='nucleophile',name='pyrrolidine',SMILES='C1CCNC1')]
        public['process_interpretation']='STY measures mass of desired product per reactor volume per time; increased conversion alone does not establish increased STY.'
        public['unknown_fixed_details']=['do not infer or request simulator kinetic constants']
    else:
        public['process_interpretation']='Analyze filling, draw, transport and injection timing. Physical volumes and pump settings are decision variables; the objective is detector peak area.'
        public['unknown_fixed_details']=['analyte identity','column chemistry','detector response model']
        public['knowledge_limit']='Only process hypotheses are supported; do not invent analytes, reagents or chemical transformations.'
    public['reasoning_requirements']=[
        'State a conditional, falsifiable chemical or process relationship using meaningful identities and physical units.',
        'Separate general mechanistic expectations from associations supported by current own observations.',
        'A source molecular identity is not a source performance claim. Do not recall published benchmark results.',
        'Candidate IDs identify transport records only. Use supplied full physical conditions to reason.']
    return public
