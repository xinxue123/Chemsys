"""One same-prefix retry; expose only redacted error classification."""
from pathlib import Path
from unittest.mock import patch
import sys,os,json,re
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
os.chdir(ROOT/'experiments/chemist_system')
from paper_v12 import study
from chem.benchmarks import knowledge_openai_client as api

def main():
    original=api.APIClient.__init__;config=api.load_config(ROOT/'openai-key.txt');errors=[]
    def record(response):
        if response.status_code<400:return
        response.read()
        try:payload=response.json();e=payload.get('error',{})
        except ValueError:e={}
        if not isinstance(e,dict):e={}
        message=str(e.get('message',''))[:600].replace(config.api_key,'[REDACTED]')
        message=re.sub(r'https?://\S+|sk-\S+|[A-Za-z0-9_\-]{24,}|[\w.+-]+@[\w.-]+','[REDACTED]',message)
        row=dict(status=response.status_code,error_type=str(e.get('type',''))[:80],error_code=str(e.get('code',''))[:80],message=message,retry_after=response.headers.get('retry-after'))
        errors.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    def init(self,*a,**kw):
        original(self,*a,**kw);self.http.event_hooks['response']=[record]
    try:
        with patch.object(api.APIClient,'__init__',init):study.run_one(ROOT,'Reizman_Suzuki_Case_1',0,'memory')
    finally:
        study.write(ROOT/'experiments/reports/paper_v12/transport_diagnostic.json',dict(single_worker=True,errors=errors,credentials_logged=False))
if __name__=='__main__':main()
