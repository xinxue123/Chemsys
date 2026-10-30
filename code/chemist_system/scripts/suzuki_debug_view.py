"""Human-readable observation layer; its oracle annotations never enter a policy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=plain) + '\n', encoding='utf8')


def csv_table(path, table, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([plain(row[k]) if isinstance(row[k], np.generic) else row[k] for k in columns] for row in table)


TRUTH = 'oracle_yield_percent_DEBUG_ONLY'
COMMON = ['candidate_id', 'catalyst', 't_res_s', 'temperature_C', 'catalyst_loading_mol_percent', TRUTH]
STAGES = [
    {'id':'gp','title':'1 · GPR均值与不确定性','file':'01_GPR.csv',
     'columns':COMMON+['GP_mean_percent','GP_latent_SD_pp','GP_observation_SD_pp','GP_mean_rank_available','available'],
     'sort':'GP_mean_rank_available',
     'rule':'根据已观察实验拟合Mixed GP。均值排名仅用于对照，尚未用知识引导。模拟真实产率只作调试对照。'},
    {'id':'base','title':'2 · 基础qLogEI与预算门限','file':'02_base_qLogEI_and_budget.csv',
     'columns':COMMON+['GP_mean_percent','GP_latent_SD_pp','base_qLogEI','base_rank_available','base_qLogEI_loss','budget_logEI','available','eligible','budget_reason'],
     'sort':'base_rank_available',
     'rule':'未查询且 max(base_qLogEI)−base_qLogEI(x) ≤ ε_t 的候选进入预算集。ε_t=log(4)×(16−t)/15；不按真实产率筛选。'},
    {'id':'region','title':'4 · 区域知识修正后的排序','file':'03_region_guidance.csv',
     'columns':COMMON+['GP_mean_percent','matches_all_knowledge','matches_retained_region','raw_region_shift_pp','mean_after_region_percent','base_qLogEI','region_unscaled_qLogEI','base_rank_available','region_rank_available','region_rank_change_from_base','eligible'],
     'sort':'region_rank_available',
     'rule':'B4保留的知识构成区域R；官方LGBO偏移d_R(x)加到GP均值，重算qLogEI。这是α=1的区域引导对照，尚未缩放KF幅度。区域条件是软引导。'},
    {'id':'kf','title':'5 · KF幅度缩放与最终排序','file':'04_KF_ranking_and_choice.csv',
     'columns':COMMON+['GP_mean_percent','raw_region_shift_pp','KF_amplitude','deployed_shift_pp','mean_after_KF_percent','base_qLogEI','region_unscaled_qLogEI','guided_qLogEI','region_rank_available','guided_rank_available','KF_rank_change_from_region','guided_rank_eligible','eligible','is_selected','reason'],
     'sort':'guided_rank_eligible',
     'rule':'使用此前反馈得到的α=clip(a,0,1)，μ_used=μ_GP+α·d_R；用同一GP协方差重算qLogEI。在预算集内取最大，完全同分时按候选ID字典序。'},
]
LABELS = {
    'candidate_id':'候选ID','catalyst':'催化剂','t_res_s':'时间 / s','temperature_C':'温度 / °C',
    'catalyst_loading_mol_percent':'催化剂用量 / mol%',TRUTH:'模拟真实产率 / %（仅对照）',
    'GP_mean_percent':'GP均值 / %','GP_latent_SD_pp':'GP模型SD / pp','GP_observation_SD_pp':'含观测噪声SD / pp',
    'GP_mean_rank_available':'GP均值排名','available':'未查询','eligible':'进入预算集',
    'base_qLogEI':'基础qLogEI','base_rank_available':'基础排名','base_qLogEI_loss':'基础分数损失',
    'budget_logEI':'预算ε','budget_reason':'预算判断','matches_all_knowledge':'满足全部知识',
    'matches_retained_region':'满足保留区域','raw_region_shift_pp':'区域原偏移 / pp',
    'mean_after_region_percent':'区域修正均值 / %','region_unscaled_qLogEI':'区域qLogEI（α=1）',
    'region_rank_available':'区域排名','region_rank_change_from_base':'相对基础提升名次',
    'KF_amplitude':'本轮KF幅度α','deployed_shift_pp':'KF后偏移 / pp','mean_after_KF_percent':'最终均值 / %',
    'guided_qLogEI':'最终qLogEI','guided_rank_available':'最终全体未查询排名',
    'KF_rank_change_from_region':'KF相对区域提升名次','guided_rank_eligible':'预算内最终排名',
    'is_selected':'实际选择','reason':'最终判断',
}
REASONS = {'already_queried':'已查询，排除','outside_base_qLogEI_budget':'基础分数损失超过门限',
           'eligible_lower_guided_score':'预算内，最终分数较低','eligible_tie_larger_candidate_id':'完全同分，ID字典序落后',
           'selected':'预算内最终分数最大，选中'}
AXES = {0:'停留时间',1:'温度',2:'催化剂用量',3:'催化剂类别'}


def annotate(table, pool, oracle_values):
    """Add display-only columns to a COPY, without mutating cache/state/decision."""
    columns = {name:table[name] for name in table.dtype.names}
    columns.update({
        TRUTH: np.asarray(oracle_values, dtype=float)*100,
        'catalyst': np.asarray([p['catalyst'] for p in pool['points']]),
        't_res_s': np.asarray([p['t_res'] for p in pool['points']]),
        'temperature_C': np.asarray([p['temperature'] for p in pool['points']]),
        'catalyst_loading_mol_percent': np.asarray([p['catalyst_loading'] for p in pool['points']]),
    })
    def ranks(values):
        result=np.zeros(len(table),dtype=np.int32);ix=np.flatnonzero(table.available)
        order=ix[np.lexsort((table.candidate_id[ix],-values[ix]))]
        result[order]=np.arange(1,len(order)+1)
        return result
    columns['GP_mean_rank_available']=ranks(table.GP_mean_percent)
    columns['region_rank_available']=ranks(table.region_unscaled_qLogEI)
    columns['region_rank_change_from_base']=table.base_rank_available-columns['region_rank_available']
    columns['KF_rank_change_from_region']=columns['region_rank_available']-table.guided_rank_available
    columns['budget_reason']=np.where(~table.available,'已查询，排除',np.where(table.eligible,'基础分数损失≤门限，入预算集','基础分数损失>门限，排除'))
    return np.rec.fromarrays(list(columns.values()), names=list(columns))


def filter_preview(bank, prior, iteration, raw_shift_y, mu_y, predictive_sd_y, y):
    """Independent arithmetic for inspection; the native update remains authoritative."""
    H=np.array([1.,raw_shift_y]);theta=np.asarray(prior['mean']);P=np.asarray(prior['covariance'])
    R=float(predictive_sd_y**2+.0009);residual=float(y-mu_y)
    innovation=float(residual-H@theta);S=float(H@P@H+R);K=P@H/S
    after=theta+K*innovation;A=np.eye(2)-np.outer(K,H)
    covariance=A@P@A.T+R*np.outer(K,K);covariance=(covariance+covariance.T)/2
    return {'enabled':True,'iteration':iteration,'units':'产率使用0..1；乘100转换为百分产率/pp',
            'state_names':['background_offset_b','regional_amplitude_a'],
            'bank_before':bank,'prior':prior,'theta_prior':theta.tolist(),'P_prior':P.tolist(),
            'H':H.tolist(),'GP_prequery_mu':float(mu_y),'observed_y':float(y),'residual_y_minus_mu':residual,
            'observation_variance':float(predictive_sd_y**2),'noise_floor_variance':.0009,'R':R,
            'predicted_residual_H_theta':float(H@theta),'innovation':innovation,'S':S,'K':K.tolist(),
            'theta_after_preview':after.tolist(),'P_after_preview':covariance.tolist(),
            'deployed_alpha_this_round':prior['amplitude'],'clipped_alpha_for_future_same_region':float(np.clip(after[1],0,1)),
            'note':'本轮选择使用更新前α；以下反馈仅供后续相同区域使用。b不加入采集均值；P不额外加入GP方差。'}


class DebugBook:
    """One clean log, one viewer, one numbered folder per actual query."""
    def __init__(self, root, options, oracle_values, pool):
        self.root,self.options,self.oracle_values,self.pool=root,options,oracle_values,pool
        self.entries=[];self.messages=[];self.current={};self.directory=None;self.tables={};self.current_table=None
        self.round_log=[];self.on_stage=None
        source=Path(__file__).with_name('suzuki_debug_viewer.html')
        (root/'index.html').write_text(source.read_text(encoding='utf8'),encoding='utf8')
        self.manifest()

    def manifest(self):
        (self.root/'rounds.js').write_text('window.DEBUG_ROUNDS='+json.dumps(self.entries,ensure_ascii=False)+';\n',encoding='utf8')

    def line(self, text=''):
        print(text,flush=True)
        with (self.root/'run.log').open('a',encoding='utf8') as f:f.write(text+'\n')
        if self.directory is not None:
            with (self.directory/'decision.log').open('a',encoding='utf8') as f:f.write(text+'\n')
            self.round_log.append(text)

    def begin(self, state):
        t=len(state['decisions'])+1
        self.directory=self.root/'runs'/state['run_id']/f'round{t:02d}'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.tables={};self.messages=[];self.round_log=[];self.current_table=None
        self.current={'case':state['case'],'regime':state['regime'],'method':state['method'],'round':t,
                      'history_before_query':deepcopy_json(state['history']),'initial_knowledge':state['initial_knowledge'],
                      'knowledge_phases':[],'llm':{'called':False,'mode':self.options.llm,'attempts':[]},'kf':None,
                      'completed':False,'csv_enabled':not self.options.no_stage_csv}
        self.line(f"\n【Case {state['case']} | {state['regime']} | {state['method']} | 追加第{t:02d}轮 / 总第{t+4}次】")
        self.line('  模拟真实产率列仅供对照；不参与评分、筛选或LLM提示。')

    def preview(self, table, rank, label):
        limit=self.options.top
        ix=np.flatnonzero((table[rank]>0)&(table[rank]<=limit))
        ix=ix[np.argsort(table[rank][ix])]
        compact=[]
        score={'GP_mean_rank_available':'GP_mean_percent','base_rank_available':'base_qLogEI',
               'region_rank_available':'region_unscaled_qLogEI','guided_rank_eligible':'guided_qLogEI'}[rank]
        for i in ix:
            compact.append(f"{table.candidate_id[i]} {label}={table[score][i]:.6f} 真={table[TRUTH][i]:.3f}%")
        if compact:self.line('  前'+str(limit)+'：'+' | '.join(compact))

    def table(self, stage_id, table):
        spec=next(s for s in STAGES if s['id']==stage_id)
        self.tables[stage_id]=spec
        if not self.options.no_stage_csv:
            columns=spec['columns'];rank=spec['sort'];valid=table[rank]>0
            order=np.lexsort((table.candidate_id,np.where(valid,table[rank],10**9)))
            csv_table(self.directory/spec['file'],table[order],columns)

    def checkpoint(self, stage_id, table):
        self.current_table=table
        self.publish()
        if self.on_stage is not None:self.on_stage(stage_id,table)

    def base_stages(self, table, decision, fit, state):
        self.current['normalization']={'y_mean':float(fit['mean']),'y_std':float(fit['std']),'best_std':fit['best_std']}
        self.current['GP_parameters_raw']={k:v.detach().cpu().tolist() for k,v in fit['model'].named_parameters()}
        self.current['base']={'reference_id':self.pool['ids'][decision['reference']],
            'max_qLogEI':float(decision['base'][decision['reference']]),'epsilon':decision['budget'],
            'cutoff_qLogEI':float(decision['base'][decision['reference']]-decision['budget']),
            'available':int(table.available.sum()),'eligible':int(table.eligible.sum())}
        self.line(f"[1 GPR] 已观察{len(state['history'])}点，得到{len(table)}个候选的均值与两种SD。表：01_GPR.csv")
        self.preview(table,'GP_mean_rank_available','μ')
        self.table('gp',table)
        self.checkpoint('gp',table)
        b=self.current['base']
        self.line(f"[2 基础qLogEI/门限] 最大={b['max_qLogEI']:.9f}（{b['reference_id']}）；ε={b['epsilon']:.9f}。")
        self.line(f"  保留：未查询 且 base_qLogEI≥{b['cutoff_qLogEI']:.9f}；{b['available']}个未查询 → {b['eligible']}个预算候选。表：02_base_qLogEI_and_budget.csv")
        self.preview(table,'base_rank_available','qLogEI')
        self.table('base',table)
        self.checkpoint('base',table)

    def knowledge(self, state, decision, phase, will_refresh):
        clauses=state['knowledge']['clauses'];ids=set(decision['retained'])
        self.current['knowledge_phases'].append({'phase':phase,'knowledge':deepcopy_json(state['knowledge']),
            'B4':decision['diagnosis'].to_dict(),'retained_ids':decision['retained'],'will_refresh':will_refresh})
        use_b4=state['method'] in ('G2_gate_KF_B4','G3_gate_KF_B4_event')
        self.line('[3 知识/B4] '+('刷新后重算。' if phase>1 else '使用当前已有知识。')+
                  ('应用B4保留子集。' if use_b4 else '本组不应用B4撤销。'))
        for h in clauses:
            value=('类别集合'+str(h['categories'])) if h['coordinate']==3 else ('归一化区间'+str(h['bounds']))
            self.line(f"  {h['id']} {AXES[h['coordinate']]} {value} → "+('用于注入' if h['id'] in ids else 'B4本轮撤下'))
        if not clauses:self.line('  当前无有效知识；区域偏移为0。')
        self.line(f"  B4诊断={decision['diagnosis'].conflict_kind}；保留={decision['retained']}。完整子集检查见页面「知识/B4」。")
        if phase==1 and not will_refresh:
            reason='固定初始知识，第一轮不重新调用LLM' if self.current['round']==1 else '事件未触发或本组不启用刷新，沿用已有知识'
            self.line('  LLM：本轮未调用。'+reason+'。')

    def final_ranking(self, table, decision):
        self.current_table=table
        self.current['bank_before']=decision['bank'];self.current['bank_prior']=decision['prior']
        self.current['bank_key']=decision['bank_key'];self.current['amplitude']=decision['amplitude']
        self.line('[4 区域修正] μ_region=μ_GP+d_R；重新计算α=1的qLogEI。表：03_region_guidance.csv')
        self.preview(table,'region_rank_available','qLogEI')
        self.table('region',table)
        self.checkpoint('region',table)
        self.line(f"[5 KF幅度/最终排序] α={decision['amplitude']:.9f}；μ_used=μ_GP+α·d_R。表：04_KF_ranking_and_choice.csv")
        if decision['bank'] is None:self.line('  本组未启用KF或当前没有区域；不更新区域校准。')
        else:self.line(f"  该区域此前更新{decision['bank']['updates']}次；均值状态[b,a]={decision['prior']['mean']}。本轮尚未使用新的实验结果。")
        self.preview(table,'guided_rank_eligible','qLogEI')
        self.table('kf',table)
        j=decision['selected'];self.current['selection']=record(table[j])
        self.line(f"[6 选定] {table.candidate_id[j]}；预算内排名1，基础损失={table.base_qLogEI_loss[j]:.9f}≤ε={decision['budget']:.9f}。")
        self.line(f"  μ_GP={table.GP_mean_percent[j]:.6f}% → 加区域{table.raw_region_shift_pp[j]:+.6f}pp → KF后μ={table.mean_after_KF_percent[j]:.6f}%；qLogEI {table.base_qLogEI[j]:.9f} → {table.region_unscaled_qLogEI[j]:.9f} → {table.guided_qLogEI[j]:.9f}。")
        self.checkpoint('kf',table)

    def llm_start(self, context, messages=None):
        reasons=context['public']['reasons'];self.current['llm'].update(called=True,reasons=reasons,context_sha256=context['context_sha256'])
        self.line('[3 LLM刷新] '+('精确上下文重放原响应' if self.options.llm=='replay' else '真实请求deepseek-flash')+'；触发='+', '.join(reasons)+'。')
        if messages is not None:
            self.current['llm']['pending_messages']=messages
            prompt='\n\n'.join(f"===== {m['role'].upper()} =====\n{m['content']}" for m in messages)
            (self.directory/'LLM_request_prompt.txt').write_text(prompt+'\n',encoding='utf8')
            self.line('  请求前完整提示词已保存：LLM_request_prompt.txt；可在debug_llm_request断点查看。')
        self.publish()

    def llm_finish(self, folder, valid=None, error=None):
        attempts=[]
        for path in sorted(folder.glob('attempt*/request.json')):
            body=json.loads(path.read_text(encoding='utf8'));attempt=path.parent.name
            response_path=path.parent/'response.json'
            response=json.loads(response_path.read_text(encoding='utf8')) if response_path.exists() else None
            messages=body['messages'];prompt='\n\n'.join(f"===== {m['role'].upper()} =====\n{m['content']}" for m in messages)
            prompt_path=self.directory/f'LLM_{attempt}_prompt.txt';prompt_path.write_text(prompt+'\n',encoding='utf8')
            content=response['choices'][0]['message'].get('content','') if response and response.get('choices') else ''
            (self.directory/f'LLM_{attempt}_response.txt').write_text(content+'\n',encoding='utf8')
            attempts.append({'attempt':attempt,'messages':messages,'response_content':content,
                             'prompt_file':prompt_path.name,'response_file':f'LLM_{attempt}_response.txt',
                             'request_parameters':{k:v for k,v in body.items() if k!='messages'}})
            if self.options.print_llm:
                self.line(prompt);self.line('===== LLM RESPONSE =====\n'+content)
        self.current['llm']['attempts']=attempts
        if valid is not None:
            self.current['llm']['plan']=valid['plan'];self.current['llm']['accepted_attempt']=valid['accepted_attempt']
            self.line('  响应摘要：'+valid['plan']['summary'][:240])
            for a in valid['plan']['actions']:
                self.line(f"  {a['id']} {a['operation']}，依据观测{a['support']}：{a['reason'][:180]}")
        else:
            self.current['llm']['error']=str(error);self.line('  本次刷新失败，保留原知识。')
        self.line('  完整提示词与原响应：页面「知识/LLM」或本轮LLM_attempt*_prompt/response.txt。')
        self.publish()

    def kf_before(self, info):
        self.current['kf']=info
        self.line(f"[7 实验反馈/KF更新前] y={info['observed_y']*100:.6f}%，实验前μ={info['GP_prequery_mu']*100:.6f}%，残差={info['residual_y_minus_mu']*100:+.6f}pp。")
        self.line(f"  H={info['H']}；R=σ_obs²+0.0009={info['R']:.9f}；创新={info['innovation']:.9f}，S={info['S']:.9f}，K={info['K']}。")
        self.line('  KF断点：查看kf_info、bank、prior；实际更新尚未执行。')
        dump(self.directory/'05_KF_update.json',info)
        self.publish()

    def kf_after(self, after, measurement):
        info=self.current['kf'];info['bank_after_native']=after;info['measurement_native']=measurement
        np.testing.assert_allclose(info['theta_after_preview'],after['mean'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(info['P_after_preview'],after['covariance'],rtol=0,atol=1e-14)
        dump(self.directory/'05_KF_update.json',info)
        self.line(f"  更新后[b,a]={after['mean']}；未来同区域部署α={info['clipped_alpha_for_future_same_region']:.9f}。本轮选择保持更新前α。")

    def finish(self, state):
        event=state['decisions'][-1]
        if self.current['kf'] is None:
            self.current['kf']={'enabled':False,'note':'本轮无KF更新。','measurement':event['measurement']}
            self.line(f"[7 实验反馈] y={event['measurement']['observed_y']*100:.6f}%；本轮无KF更新。")
        self.current['event']=event;self.current['completed']=True
        self.current['incumbent_percent']=max(h['y'] for h in state['history'])*100
        self.line(f"[轮次完成] 当前最佳={self.current['incumbent_percent']:.6f}%；全部表格、提示词和KF见 index.html。")
        self.line('  本轮目录：'+str(self.directory.relative_to(self.root)))
        self.current['log']=list(self.round_log)
        self.publish()

    def publish(self):
        if self.directory is None:return
        table=self.current_table
        if table is not None:
            self.current['candidate_columns']={name:table[name].tolist() for name in table.dtype.names}
        self.current['stages']=[s for s in STAGES if s['id'] in self.tables]
        self.current['labels']=LABELS;self.current['reason_labels']=REASONS
        raw=json.dumps(self.current,ensure_ascii=False,default=plain,separators=(',',':'))
        (self.directory/'view.js').write_text('window.showDebugRound('+raw+');\n',encoding='utf8')
        meta={k:v for k,v in self.current.items() if k!='candidate_columns'}
        dump(self.directory/'steps.json',meta)
        entry={'case':self.current['case'],'regime':self.current['regime'],'method':self.current['method'],
               'round':self.current['round'],'completed':self.current['completed'],
               'path':(self.directory/'view.js').relative_to(self.root).as_posix(),
               'directory':self.directory.relative_to(self.root).as_posix()}
        self.entries=[v for v in self.entries if v['path']!=entry['path']]+[entry]
        self.manifest()


def deepcopy_json(value):
    return json.loads(json.dumps(value,ensure_ascii=False,default=plain))


def record(row):
    return {k:plain(row[k]) if isinstance(row[k],np.generic) else row[k] for k in row.dtype.names}
