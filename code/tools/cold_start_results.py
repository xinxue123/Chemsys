"""Audit cold-start evidence and separately disclose transport-format clarifiers.

This helper cannot launch or choose an experiment. Scientific sources stay
frozen. Additional collaborator messages resolve only the pre-existing bare
Reasoning BO Insight schema; raw responses are never edited by this helper.
"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/experiment_tools'),str(ROOT/'code/chemist_system')]
from cold_start_v13 import study,transport,validation

INITIAL_MESSAGE = ('Schema-only clarification for this Reasoning BO request: the top-level output is exactly '
    'comment, keywords, hypotheses. The V13-only knowledge_uses and assessments clauses do not apply '
    'to this bare official Insight. Follow the original scientific selection and rationale rules; '
    'no other instruction changes.')


def record_initial(agent):
    found=[r for r in transport.pending(ROOT) if '/root/'+r['invocation']['task_name']==agent]
    if len(found)!=1:
        raise ValueError('One actual pending request required')
    path=Path(found[0]['request'])
    req=transport.verify_request(path)
    if req['query']!=1 or not req['task'].endswith('_rbo') or req['attempt']!=1:
        raise ValueError('Only first-round, first-attempt Reasoning BO schema clarification')
    transport.write_once(path.parent/'format_clarification.json',dict(agent_id=agent,
        message=INITIAL_MESSAGE,kind='schema_clarification_before_first_response',
        request_sha256=req['request_sha256'],response_bytes_edited_by_parent=False))
    print('Schema-only collaborator message recorded.')


def audit():
    result=validation.audit(ROOT)
    rows=[]
    for path in (study.directory(ROOT)/'requests').glob('*/*/attempt*/format_clarification.json'):
        note=transport.read(path)
        req=transport.verify_request(path.parent/'request.json')
        raw_path=path.parent/'response.raw.json'
        if not raw_path.exists():
            continue
        raw=transport.read(raw_path)
        receipt=transport.read(path.parent/'response_receipt.json')
        if note['request_sha256']!=req['request_sha256'] or note['agent_id']!=receipt['agent_id'] or note['response_bytes_edited_by_parent']:
            raise ValueError('Schema clarification custody mismatch')
        if note['kind']=='schema_only_projection_of_prior_response':
            previous=req['public']['original_response']
            if raw!={k:previous[k] for k in ('comment','keywords','hypotheses')}:
                raise ValueError('Schema repair changed scientific fields')
            if not note['selected_point_equal_first_attempt']:
                raise ValueError('First valid scientific choice changed during format-only repair')
        elif note['message']!=INITIAL_MESSAGE:
            raise ValueError('Unexpected schema clarification')
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=transport.file_sha(path),
            agent_id=note['agent_id'],kind=note['kind'],raw_response_sha256=transport.file_sha(raw_path),
            no_parent_response_edit=True))
    result['schema_only_collaborator_clarifications']=rows
    result['format_adapter_source_sha256']=transport.file_sha(__file__)
    output=ROOT/'experiments/reports'/study.VARIANT
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'report_zh.txt').open('a',encoding='utf-8') as f:
        f.write(f'\n格式说明：{len(rows)}条请求另行记录了仅限JSON字段的协作者澄清。Reasoning BO首轮存在V13注释字段歧义；首例已验证删除多余字段前后原有科学字段与首选候选保持一致。原始响应和失败尝试均保留，未因此增加实验查询。\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='record':
        record_initial(sys.argv[2])
    else:
        audit()
