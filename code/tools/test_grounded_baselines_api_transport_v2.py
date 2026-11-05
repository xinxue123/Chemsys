"""Targeted transport-custody tests; no network, GP fit or oracle read."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import grounded_baselines_api_dispatch_v2 as dispatch


class TransportCustody(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=dispatch.ROOT/'.tmp')
        self.root = Path(self.temp.name)
        self.af = self.root/'requests/logical/context/attempt1'
        self.af.mkdir(parents=True)
        self.reg = self.root/'registration.json'
        self.registration = dict(model='gpt-6.1-terra',effort='medium',wire_parameters=dict(
            model='gpt-6.1-terra',reasoning={'effort':'medium'},store=False,max_output_tokens=8192,
            text={'format':{'type':'json_object'}},tools=[],previous_response_id=None))
        dispatch.write_once(self.reg,self.registration)
        self.req = dict(system='Return one JSON object.',public={'own_observations':[]},
                        model='gpt-6.1-terra',effort='medium',request_sha256='test-request')
        dispatch.write_once(self.af/'request.json',self.req)
        dispatch.write_once(self.af/'invocation.json',dict(model='gpt-6.1-terra',reasoning_effort='medium'))
        self.ip,self.op = self.root/'io/input.json',self.root/'io/output.json'
        dispatch.write_once(self.ip,dict(system=self.req['system'],public=self.req['public']))
        dispatch.write_once(self.af/'export.json',dict(input='io/input.json',output='io/output.json',input_sha256=dispatch.sha(self.ip)))
        self.config = type('Config',(),{'api_key':'unit-test-placeholder'})()
        self.patches = [patch.object(dispatch,'ROOT',self.root),patch.object(dispatch,'REG',self.reg)]
        for p in self.patches:p.start()
        self.calls = []
        owner = self
        class Client:
            def __init__(self,*a,**k):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def complete(self,system,user,**kwargs):
                owner.calls.append((system,user,kwargs))
                return dict(output_text='{"answer":"test"}',id='resp_unit_test',model='gpt-6.1-terra',
                    requested_model='gpt-6.1-terra',resolved_model='gpt-6.1-terra',status='completed',usage={'input_tokens':1,'output_tokens':2})
        self.client = Client

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def test_seal_exact_input_and_idempotent_resume(self):
        dispatch.serve(self.af,self.config,self.registration,self.client)
        receipt = dispatch.read(self.af/'receipt.json')
        self.assertEqual(receipt['raw_sha256'],dispatch.sha(self.af/'response.raw.json'))
        self.assertEqual(receipt['completion_sha256'],dispatch.sha(self.af/'completion.json'))
        self.assertEqual(dispatch.read(self.op),{'answer':'test'})
        self.assertEqual(self.calls,[(self.req['system'],self.req['public'],{'json_output':True,'max_output_tokens':8192})])
        self.assertEqual(dispatch.serve(self.af,self.config,self.registration,self.client)['status'],'ALREADY_SEALED')
        self.assertEqual(len(self.calls),1)

    def test_plain_text_transport_preserves_non_JSON_scientific_prompt(self):
        inp = {'system':'Return exactly {\"ok\":true}.','public':{'own_observations':[]}}
        self.assertFalse(dispatch.wants_json_mode(inp))
        reg = {'wire_parameters':{'model':'gpt-6.1-terra'}}
        self.assertNotIn('text',dispatch.resolved_wire_parameters(inp,reg))
        self.assertIn('text',dispatch.resolved_wire_parameters({'system':'Return JSON.','public':{}},reg))

    def test_input_drift_fails_before_HTTP(self):
        self.ip.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(AssertionError,'Input changed'):
            dispatch.serve(self.af,self.config,self.registration,self.client)
        self.assertEqual(self.calls,[])

    def test_interrupted_call_cannot_silently_resample(self):
        class Interrupt(self.client):
            def complete(self,*a,**k):raise RuntimeError('Simulated transport interruption')
        with self.assertRaises(RuntimeError):dispatch.serve(self.af,self.config,self.registration,Interrupt)
        self.assertFalse((self.af/'receipt.json').exists())
        with self.assertRaisesRegex(AssertionError,'Interrupted HTTP call'):
            dispatch.serve(self.af,self.config,self.registration,self.client)
        self.assertEqual(self.calls,[])


if __name__ == '__main__':unittest.main()
