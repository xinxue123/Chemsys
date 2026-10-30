"""Run organized Chem code while retaining archived scientific path identities."""
from pathlib import Path
import json
import os
import runpy
import sys

WORKSPACE=Path(__file__).resolve().parents[1]
CODE=WORKSPACE/'code/chemist_system'
LEGACY=WORKSPACE/'chemist_system'
MANIFEST=WORKSPACE/'runtime/layout_20261030/migration_manifest.json'
ALIASES={'r5':'continuous_r5.py','p0':'stable_suzuki.py','p1':'stable_feedback_r2.py',
         'r4':'continuous_r4.py','r3':'continuous_r3.py','r2':'continuous_r2.py'}


def synchronize_sources():
    """Rebind an editor's atomic save to the compatibility name, never copy logs.

    Old protocols still check source hashes. Editing science invalidates their
    frozen checks just as before the move; no protocol hash is rewritten here.
    """
    global LEGACY
    if not MANIFEST.is_file():
        sys.path.insert(0,str(WORKSPACE/'code/tools'))
        from bootstrap_checkout import initialize
        initialize(WORKSPACE)
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    LEGACY=WORKSPACE/manifest.get('legacy_path','chemist_system')
    archive=(WORKSPACE/manifest['experiments_path']).resolve()
    bindings={row['legacy_relative']:row for row in manifest['source_files']}
    for base in [CODE,WORKSPACE/'code/experiment_tools']:
        for path in base.rglob('*'):
            if not path.is_file() or {'__pycache__','.git','node_modules'}.intersection(path.parts) or path.suffix in {'.pyc','.pyo'}:continue
            relative=path.relative_to(base).as_posix()
            bindings.setdefault(relative,{'legacy_relative':relative,'canonical_relative':path.relative_to(WORKSPACE).as_posix()})
    for row in bindings.values():
        canonical=(WORKSPACE/row['canonical_relative']).resolve()
        target=LEGACY/row['legacy_relative']
        if not canonical.is_relative_to((WORKSPACE/'code').resolve()) or not target.resolve().is_relative_to(archive):
            raise RuntimeError('Compatibility binding outside the organized workspace')
        if not canonical.is_file():raise FileNotFoundError(canonical)
        if target.exists() and os.path.samefile(canonical,target):continue
        # Removing one hard-link name leaves the canonical source intact.
        if target.exists():target.unlink()
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(canonical,target)


def main():
    args=sys.argv[1:]
    if not args or args[0] in {'list','--help','-h'}:
        print('Usage: run.ps1 r5 status | p0 status | p1 status --variant feedback_t0')
        print('       run.ps1 ENTRY [args] | script research/PATH.py [args] | module PACKAGE [args]')
        print('       run.ps1 tests tests.test_continuous_r5 -v')
        print('Source entries: '+', '.join(p.stem for p in sorted(CODE.glob('*.py'))))
        return
    synchronize_sources()
    os.environ.setdefault('CHEMIST_HOME',str(LEGACY))
    os.environ.setdefault('CHEMIST_DATA',str(LEGACY/'data'))
    # Keep this lexical path: archived protocols contain these __file__ names.
    sys.path.insert(0,str(LEGACY))
    os.chdir(LEGACY)
    command=args.pop(0)
    if command=='tests':
        sys.argv=['unittest',*args];runpy.run_module('unittest',run_name='__main__',alter_sys=True);return
    if command=='module':
        if not args:raise SystemExit('module needs an importable package name')
        module=args.pop(0);sys.argv=[module,*args]
        runpy.run_module(module,run_name='__main__',alter_sys=True);return
    if command=='script':
        if not args:raise SystemExit('script needs an original project-relative path')
        relative=args.pop(0)
    else:relative=ALIASES.get(command,command if command.endswith('.py') else command+'.py')
    script=LEGACY/relative
    if not script.resolve().is_relative_to((WORKSPACE/'experiments/chemist_system').resolve()):
        raise SystemExit('Script must be in the project compatibility view')
    if not script.is_file():raise SystemExit(f'Unknown entry: {relative}')
    if (command in ALIASES or Path(relative).name.startswith('continuous_') or Path(relative).name=='stable_suzuki.py') and '--config' not in args:
        args.extend(['--config',str(WORKSPACE/'config.txt')])
    sys.argv=[str(script),*args]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
