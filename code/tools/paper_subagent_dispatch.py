"""File custody and CPU-worker helpers; model spawning uses collaboration tools."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12_subagent import study,transport

def alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        result=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(result))
        return bool(ok and result.value==259)
    finally:kernel.CloseHandle(ctypes.c_void_p(handle))

def launch(job):
    study.check(ROOT);base=study.base(ROOT,*job);sf=study.run_path(ROOT,*job)/'state.json'
    if sf.exists() and transport.read(sf)['status']=='COMPLETE':return dict(job=list(job),status='COMPLETE')
    workers=base/'workers'
    if workers.exists():
        for path in workers.glob('*/process.json'):
            r=transport.read(path)
            if alive(r['pid']):return dict(job=list(job),status='RUNNING',pid=r['pid'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=workers/stamp;folder.mkdir(parents=True)
    cmd=[str(ROOT/'runtime/environments/.venv-official/Scripts/python.exe'),'-u',
         str(ROOT/'code/experiment_tools/paper_v12_subagent'),'--root',str(ROOT),'--job',job[0],str(job[1]),job[2]]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (folder/'stdout.log').open('wb') as out,(folder/'stderr.log').open('wb') as err:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    r=dict(job=list(job),pid=process.pid,created_utc=datetime.now(timezone.utc).isoformat(),command=cmd,
           stdout=(folder/'stdout.log').relative_to(ROOT).as_posix(),stderr=(folder/'stderr.log').relative_to(ROOT).as_posix())
    transport.write_once(folder/'process.json',r)
    # Keep the exec session alive: Windows job objects reap detached descendants
    # when their short-lived launcher exits. The CLI tool yields a live session.
    print(json.dumps(dict(r,status='STARTED')),flush=True)
    code=process.wait()
    transport.write_once(folder/'exit.json',dict(returncode=code,finished_utc=datetime.now(timezone.utc).isoformat()))
    return dict(r,status='FINISHED' if code==0 else 'FAILED',returncode=code)

def prepare(request,task_name):
    p=study.check(ROOT);request=Path(request).resolve();export=transport.export_request(ROOT,request)
    template=ROOT/p['decision_instructions_path']
    if transport.file_sha(template)!=p['decision_instructions_sha256']:raise ValueError('Instructions changed')
    message=template.read_text(encoding='utf-8').format(input_path=(ROOT/export['packet_relpath']).as_posix(),output_path=(ROOT/export['output_relpath']).as_posix())
    invocation=dict(task_name=task_name,fork_turns='none',model=p['model'],reasoning_effort=p['reasoning_effort'],message=message)
    transport.write_once(request.parent/'spawn_request.json',invocation)
    return invocation

def record_spawn(request,result):
    request=Path(request).resolve();invocation=transport.read(request.parent/'spawn_request.json')
    agent=result.get('task_name',result.get('agent_id'))
    transport.write_once(request.parent/'spawn_result.json',result)
    return transport.assign(request,agent,dict(spawn_request=invocation,spawn_result=result))

def submit(request):
    request=Path(request).resolve();export=transport.read(request.parent/'export.json');assigned=transport.read(request.parent/'assignment.json')
    return transport.submit(request,ROOT/export['output_relpath'],assigned['agent_id'])

def progress():
    rows=study.status(ROOT);pending=study.pending(ROOT)
    return dict(complete=sum(r['status']=='COMPLETE' for r in rows),total=len(rows),
                queries=sum(r['queries'] for r in rows),rows=[r for r in rows if r['status']!='NOT_STARTED'],pending=pending)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('launch','prepare','record','submit','progress'))
    ap.add_argument('--job',nargs=3);ap.add_argument('--request',type=Path);ap.add_argument('--task-name');ap.add_argument('--result-json');ap.add_argument('--result-file',type=Path)
    a=ap.parse_args()
    if a.action=='launch':out=launch((a.job[0],int(a.job[1]),a.job[2]))
    elif a.action=='prepare':out=prepare(a.request,a.task_name)
    elif a.action=='record':out=record_spawn(a.request,transport.read(a.result_file) if a.result_file else json.loads(a.result_json))
    elif a.action=='submit':out=submit(a.request)
    else:out=progress()
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
