import json
import numpy as np
import pytest
from chem.benchmarks import deepseek_suzuki_client as c


def valid_card():return {'bias':[0]*8,'target':[.5]*3,'curvature':[10]*3,'novelty':5,'anchors':[],'reason':'compare unresolved regions from observation1','support':[1]}


def response(card,finish='stop'):return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(card)}}]}


def test_config_plaintext_keeps_secret_out_of_repr(tmp_path):
    p=tmp_path/'test_config.txt';p.write_text('https://api.deepseek.com\nDEEPSEEK_API_KEY=fake-secret-for-test\n',encoding='utf8')
    cfg=c.load_config(p);assert cfg.api_key=='fake-secret-for-test'
    assert 'fake-secret-for-test' not in repr(cfg) and 'fake-secret-for-test' not in json.dumps(cfg.public())


def test_config_rejects_url_credentials(tmp_path):
    p=tmp_path/'test_config.txt';p.write_text('https://user:pass@example.com\nDEEPSEEK_API_KEY=fake\n',encoding='utf8')
    with pytest.raises(ValueError):c.load_config(p)


def test_secret_redaction():
    cfg=c.APIConfig('https://api.deepseek.com','test-secret')
    assert 'test-secret' not in str(c.safe_error(ValueError('bad test-secret'),cfg))


def test_valid_scorecard_and_future_reference_rejected():
    card=valid_card();assert c.parse_card(response(card),[{}])==card
    card['support']=[2]
    with pytest.raises(ValueError):c.parse_card(response(card),[{}])


def test_truncated_and_empty_responses_never_fallback_to_a_choice():
    with pytest.raises(ValueError):c.parse_card(response(valid_card(),'length'),[{}])
    with pytest.raises(ValueError):c.parse_card({'choices':[{'finish_reason':'stop','message':{'content':''}}]},[{}])


def test_context_is_only_this_request():
    request={'task':'opaque','observations':[{'index':1,'yield_percent':35.}],'previous_card':None}
    messages=c.messages_for(request)
    assert len(messages)==2 and json.loads(messages[1]['content'])==request
    assert c.PARAMETERS['model']=='deepseek-flash' and c.PARAMETERS['thinking']['type']=='enabled'


def test_boolean_and_nonfinite_numeric_parameters_rejected():
    card=valid_card();card['bias'][0]=True
    with pytest.raises(ValueError):c.parse_card(response(card),[{}])


def test_schema_repair_then_cached_first_valid_answer(tmp_path):
    from chem.benchmarks import deepseek_agent_suzuki_study as s
    class FakeClient:
        config=c.APIConfig('https://api.deepseek.com','fake-secret')
        calls=0
        def complete(self,messages):
            self.calls+=1
            answer=response({'wrong':True} if self.calls==1 else valid_card())
            answer.update(model=c.MODEL,id=str(self.calls),system_fingerprint=None)
            return answer
    client=FakeClient();request={'public':{'task':'synthetic','observations':[{'index':1,'yield_percent':25.}]},'context_sha256':'synthetic-context'}
    result=s.get_card(client,request,[{}],tmp_path/'requests')
    assert result['accepted_attempt']==2 and client.calls==2
    assert s.get_card(client,request,[{}],tmp_path/'requests')==result and client.calls==2


def test_three_invalid_answers_fail_without_local_fallback(tmp_path):
    from chem.benchmarks import deepseek_agent_suzuki_study as s
    class FakeClient:
        config=c.APIConfig('https://api.deepseek.com','fake-secret')
        calls=0
        def complete(self,messages):
            self.calls+=1
            answer=response({'wrong':True});answer.update(model=c.MODEL,id=str(self.calls))
            return answer
    client=FakeClient();folder=tmp_path/'requests'
    with pytest.raises(RuntimeError):s.get_card(client,{'public':{'task':'synthetic'},'context_sha256':'synthetic'},[{}],folder)
    assert client.calls==3 and not (folder/'validated.json').exists()


def test_transport_keeps_only_content_metadata_and_redacts_key():
    import httpx
    cfg=c.APIConfig('https://api.deepseek.com','synthetic-secret-value')
    def handler(request):
        assert cfg.api_key not in request.content.decode()
        return httpx.Response(200,json={'id':'synthetic','model':c.MODEL,'created':0,
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'synthetic-secret-value','reasoning_content':'private test reasoning'}}],
            'usage':{'total_tokens':10}})
    client=c.APIClient(cfg);client.http.close();client.http=httpx.Client(base_url=cfg.base_url,transport=httpx.MockTransport(handler))
    try:answer=client.complete([{'role':'user','content':'synthetic'}])
    finally:client.close()
    assert cfg.api_key not in json.dumps(answer) and 'private test reasoning' not in json.dumps(answer)
    assert answer['choices'][0]['reasoning_characters']>0 and not answer['retained_raw_reasoning']
    card=valid_card();card['target'][0]=float('nan')
    with pytest.raises(ValueError):c.parse_card(response(card),[{}])
