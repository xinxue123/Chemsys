"""Secret-safe DeepSeek API transport and prefix-only scorecard contract."""
from dataclasses import dataclass,field
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx

from .agent_score_suzuki_engine import validate_card

MODEL='deepseek-flash'
PARAMETERS={'model':MODEL,'thinking':{'type':'enabled'},'reasoning_effort':'high',
            'max_tokens':8192,'response_format':{'type':'json_object'},'stream':False}
SYSTEM_PROMPT='''You are a chemistry experiment decision agent. Maximize the best observed Suzuki reaction yield within the remaining query budget. Your only experiment evidence is the observed history in this request. Do not identify a benchmark case, recall its outcomes, invent ligand structures for anonymous catalyst labels, use other trajectories, or assume access to unseen yields. You have no GP, Bayesian acquisition function, external tool, file access, or yield simulator.

You must return one JSON scorecard. A fixed numerical executor will score EVERY member of a fixed pool (4692 candidates) using your rule, exclude already queried candidate IDs, and query the highest score (lexical ID tie). You cannot directly choose an ID or create a new candidate. The three continuous coordinates z are normalized to [0,1] in the order residence time, temperature, catalyst loading. Catalyst index c is 0..7; no distance or chemistry is implied by the category ordering.

Score formula:
s(x)=bias[c] - sum_j(curvature[j]*(z[j]-target[j])**2)
     + novelty*d_same_category(z,observed_history)
     + sum_a(amplitude_a*exp(-sum_j((z[j]-observed_z[index_a][j])**2)/(2*width_a**2))*same_catalyst_as_anchor).
d_same_category is nearest observed numeric Euclidean distance divided by sqrt(3), capped at1; it equals1 if that catalyst has never been observed. This is a geometric novelty heuristic, not predictive variance. Gaussian anchors refer only to this request's observed indices (1-based). A positive amplitude favors an observed neighborhood; a negative amplitude discourages it. Utilities need not be calibrated predicted yields.

Return EXACTLY these seven JSON fields, no markdown or extra fields:
"bias": eight finite numbers in [-100,150], one per catalyst index;
"target": three finite numbers in [0,1];
"curvature": three finite nonnegative numbers in [0,150];
"novelty": a finite number in [0,50];
"anchors": zero to eight triples [observed_index, amplitude_in_-50_to_50, width_in_0.03_to_2];
"reason": a concise decision rationale grounded in observed results, not a detailed internal reasoning trace;
"support": a nonempty list of observed integer indices supporting this decision.

Each new outcome is evidence for the NEXT scorecard. Adapt or retain your rule as justified, balancing unresolved catalysts/numeric regions and incumbent refinement within the remaining budget. Use the entire observed history and previous scorecard supplied here. Do not treat a single catalyst tested in weak numeric conditions as conclusively poor. Output only the JSON object.'''


@dataclass(frozen=True)
class APIConfig:
    base_url:str
    api_key:str=field(repr=False)

    def public(self):return {'base_url':self.base_url,'model':MODEL,'credential_source':'config.txt (not copied or hashed)'}


def load_config(path):
    raw=Path(path).read_text(encoding='utf-8-sig').strip()
    try:
        parsed=json.loads(raw)
        url=parsed.get('base_url') or parsed.get('url') or parsed.get('endpoint')
        key=parsed.get('DEEPSEEK_API_KEY') or parsed.get('api_key')
    except json.JSONDecodeError:
        urls=re.findall(r'https?://[^\s\"\x27,;]+',raw);url=urls[0] if urls else None
        match=re.search(r'(?im)^\s*DEEPSEEK_API_KEY\s*[:=]\s*(.+?)\s*$',raw)
        key=match.group(1).strip().strip('\"\x27') if match else None
    if not url or not key:raise ValueError('config.txt must contain an HTTPS endpoint and DEEPSEEK_API_KEY')
    parts=urlsplit(url)
    if parts.scheme!='https' or parts.username or parts.password or parts.query or parts.fragment:raise ValueError('unsupported credential-bearing or non-HTTPS endpoint')
    return APIConfig(url.rstrip('/'),key)


def safe_error(error,config):
    # Never log HTTP headers, credentials, error response bodies, or raw traceback.
    if isinstance(error,httpx.HTTPStatusError):return {'kind':'HTTPStatusError','status':error.response.status_code}
    return {'kind':type(error).__name__,'message':str(error).replace(config.api_key,'[REDACTED]')[:250]}


class APIClient:
    def __init__(self,config):
        self.config=config
        self.http=httpx.Client(base_url=config.base_url,headers={'Authorization':'Bearer '+config.api_key},
                               timeout=httpx.Timeout(180,connect=20),follow_redirects=False)
    def close(self):self.http.close()
    def models(self):
        r=self.http.get('/models');r.raise_for_status();data=r.json()
        return {'object':data.get('object'),'data':[{k:v for k,v in m.items() if k in
            ('id','object','owned_by','name','context_window','max_output_tokens','effort','api_capabilities')} for m in data.get('data',[])]}
    def complete(self,messages):
        payload=dict(PARAMETERS,messages=messages);start=time.perf_counter()
        r=self.http.post('/chat/completions',json=payload);r.raise_for_status()
        # Only retain response content and reproducibility metadata; omit headers
        # and hidden reasoning text. Its count/hash permits audit without a dump.
        data=json.loads(r.text.replace(self.config.api_key,'[REDACTED]'));choices=[]
        for item in data.get('choices',[]):
            message=item.get('message',{});reasoning=message.get('reasoning_content') or ''
            choices.append({'index':item.get('index'),'finish_reason':item.get('finish_reason'),
                            'message':{'role':message.get('role'),'content':message.get('content')},
                            'reasoning_characters':len(reasoning),'reasoning_sha256':sha256(reasoning.encode('utf8')).hexdigest()})
        return {'id':data.get('id'),'created':data.get('created'),'model':data.get('model'),
                'system_fingerprint':data.get('system_fingerprint'),'choices':choices,'usage':data.get('usage',{}),
                'wall_seconds':time.perf_counter()-start,'retained_raw_reasoning':False}


def parse_card(response,history):
    choices=response.get('choices',[])
    if len(choices)!=1:raise ValueError('expected exactly one choice')
    if choices[0]['finish_reason']!='stop':raise ValueError('completion did not finish normally')
    content=choices[0]['message'].get('content')
    if not isinstance(content,str) or not content.strip():raise ValueError('empty JSON content')
    card=json.loads(content)
    if not isinstance(card,dict):raise ValueError('scorecard must be a JSON object')
    validate_card(card,history)
    if any(isinstance(v,bool) for k in ('bias','target','curvature') for v in card[k]):raise ValueError('boolean numeric field')
    if isinstance(card['novelty'],bool):raise ValueError('boolean novelty')
    if any(isinstance(v,bool) for anchor in card['anchors'] for v in anchor):raise ValueError('boolean anchor field')
    return card


def messages_for(request):
    return [{'role':'system','content':SYSTEM_PROMPT},
            {'role':'user','content':json.dumps(request,ensure_ascii=False,allow_nan=False,separators=(',',':'))}]
