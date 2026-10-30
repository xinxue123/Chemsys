import json
import httpx
import pytest
from chem.benchmarks import deepseek_small_suzuki_study as s
from chem.benchmarks import deepseek_suzuki_client as api


def answer(valid=True):
    card={'bias':[0]*8,'target':[.5]*3,'curvature':[10]*3,'novelty':5,'anchors':[],'reason':'observed initial result','support':[1]} if valid else {'bad':True}
    return {'id':'mock','model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps(card)}}]}


def test_reduced_parallelism_and_reasoning_cap():
    assert s.WORKERS==2 and s.PARAMETERS['max_tokens']==16384
    assert s.PARAMETERS['thinking']=={'type':'enabled'} and s.PARAMETERS['reasoning_effort']=='high'


def test_actual_transport_payload_matches_recorded_parameters():
    config=api.APIConfig('https://api.deepseek.com','synthetic-test-secret');client=api.APIClient(config);client.http.close()
    def handler(request):
        body=json.loads(request.content);assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
        assert 'synthetic-test-secret' not in request.content.decode()
        result=answer();result['choices'][0]['message']['reasoning_content']='private test reasoning';result['usage']={'total_tokens':10}
        return httpx.Response(200,json=result)
    client.http=httpx.Client(base_url=config.base_url,transport=httpx.MockTransport(handler))
    try:result=s.complete(client,[{'role':'user','content':'mock'}])
    finally:client.close()
    assert 'private test reasoning' not in json.dumps(result) and not result['retained_raw_reasoning']


def test_balance_error_sets_global_stop_without_more_calls(tmp_path,monkeypatch):
    class Fake:config=api.APIConfig('https://api.deepseek.com','fake')
    calls=[]
    def fail(*args):
        calls.append(1);r=httpx.Response(402,request=httpx.Request('POST','https://api.deepseek.com/chat/completions'));r.raise_for_status()
    monkeypatch.setattr(s,'complete',fail);s.STOP.clear()
    context={'public':{'task':'mock'},'history':[{}],'context_sha256':'mock'}
    try:
        with pytest.raises(RuntimeError):s.get_card(Fake(),context,tmp_path/'first')
        with pytest.raises(RuntimeError):s.get_card(Fake(),context,tmp_path/'second')
        assert s.STOP.is_set() and len(calls)==1
    finally:s.STOP.clear()


def test_rate_limit_does_not_consume_received_answer_budget(tmp_path,monkeypatch):
    class Fake:config=api.APIConfig('https://api.deepseek.com','fake')
    calls=[]
    def generate(*args):
        calls.append(1)
        if len(calls)==1:httpx.Response(429,request=httpx.Request('POST','https://api.deepseek.com/chat/completions')).raise_for_status()
        return answer()
    monkeypatch.setattr(s,'complete',generate);monkeypatch.setattr(s.time,'sleep',lambda _:None);s.STOP.clear()
    result=s.get_card(Fake(),{'public':{'task':'mock'},'history':[{}],'context_sha256':'mock'},tmp_path/'request')
    assert len(calls)==2 and result['accepted_attempt']==2 and result['received_responses']==1


def test_invalid_schema_never_gets_manual_fallback(tmp_path,monkeypatch):
    class Fake:config=api.APIConfig('https://api.deepseek.com','fake')
    calls=[]
    def generate(*args):calls.append(1);return answer(False)
    monkeypatch.setattr(s,'complete',generate);s.STOP.clear()
    with pytest.raises(RuntimeError):s.get_card(Fake(),{'public':{'task':'mock'},'history':[{}],'context_sha256':'mock'},tmp_path/'request')
    assert len(calls)==3 and not (tmp_path/'request/validated.json').exists()
