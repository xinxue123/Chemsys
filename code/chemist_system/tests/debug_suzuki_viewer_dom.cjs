// Offline DOM contract test. No browser process, HTTP server, network or downloads.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(process.argv[2]);
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const ids=new Map();let context;const blobs=[];
class Element{
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this._text='';this._value='';this.disabled=false;}
  set id(v){this._id=v;ids.set(v,this);}get id(){return this._id;}
  set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
  set value(v){this._value=String(v);}get value(){return this._value;}
  append(...children){for(const child of children){this.children.push(child);
    if(this.tagName==='SELECT'&&this._value==='')this._value=child.value;
    if(child.tagName==='SCRIPT'&&child.src){
      const f=path.resolve(root,child.src.split('?')[0]);
      assert(f.startsWith(root+path.sep),'scripts must stay inside the generated output');
      vm.runInContext(fs.readFileSync(f,'utf8'),context,{filename:f});
      if(child.onload)child.onload();
    }
  }}
  replaceChildren(...children){this.children=[];this._text='';if(this.tagName==='SELECT')this._value='';this.append(...children);}
  click(){if(this.onclick)this.onclick();}focus(){}setSelectionRange(){}
}
const document={createElement:tag=>new Element(tag),getElementById:id=>ids.get(id),body:new Element('body')};
for(const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)){const n=new Element(match[1]);n.id=match[2];}
context=vm.createContext({document,window:{},console,Date,Blob,URL:{createObjectURL:b=>{blobs.push(b);return'blob:offline-test';},revokeObjectURL:()=>{}},setTimeout:fn=>fn()});
vm.runInContext(fs.readFileSync(path.join(root,'rounds.js'),'utf8'),context);
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context,{filename:'viewer-inline.js'});
function walk(node){return[node,...node.children.flatMap(walk)];}
function button(text,container='content'){const n=walk(ids.get(container)).find(n=>n.tagName==='BUTTON'&&n.textContent===text);assert(n,'button missing: '+text);n.click();}
function cellText(){return ids.get('content').textContent;}
async function main(){
  assert(ids.get('title').textContent.includes('第1轮'));
  assert(cellText().includes('模拟真实产率 / %'));
  assert(cellText().includes('4692'));
  assert(walk(ids.get('content')).filter(n=>n.tagName==='TR').length===51,'one header plus 50 paginated rows');
  let field=ids.get('candidate-search');field.value='p00017';field.selectionStart=6;field.oninput();
  assert(cellText().includes('匹配1 / 4692行'));
  button('2 qLogEI / 门限','tabs');
  assert(cellText().includes('匹配1 / 4692行'),'candidate search persists across stages');
  assert(cellText().includes('基础分数损失'));
  button('清除筛选');button('5–6 KF / 最终选择','tabs');button('定位最终选中');
  assert(ids.get('candidate-search').value===ids.get('chosen').textContent);
  assert(cellText().includes('预算内最终分数最大'));
  button('清除筛选');
  const heading=walk(ids.get('content')).find(n=>n.tagName==='TH'&&n.title.startsWith('guided_qLogEI；'));
  assert(heading);heading.click();
  assert(cellText().includes('最终qLogEI ↑'),'column sort executes');
  button('下载本阶段全部4692行 CSV');
  const csv=await blobs.at(-1).text();
  assert(csv.includes('oracle_yield_percent_DEBUG_ONLY'));
  assert(csv.split('\r\n').length===4693,'CSV download includes every candidate, not just one page');
  const r3=walk(ids.get('rounds')).find(n=>n.tagName==='BUTTON'&&n.textContent==='03');assert(r3);r3.click();
  assert(ids.get('title').textContent.includes('第3轮'));
  button('3 知识 / LLM / B4','tabs');
  assert(cellText().includes('本次为精确上下文重放'));
  assert(cellText().includes('You update a preferred Suzuki reaction REGION'));
  assert(cellText().includes('完整提示词 · system')&&cellText().includes('完整提示词 · user'));
  assert(cellText().includes('LLM原始响应'));
  button('7 反馈 / KF更新','tabs');
  assert(cellText().includes('卡尔曼增益K')&&cellText().includes('实际更新后θ⁺'));
  assert(cellText().includes('本轮选择使用更新前α'));
  button('本轮日志','tabs');assert(cellText().includes('[1 GPR]')&&cellText().includes('[轮次完成]'));
  console.log(JSON.stringify({status:'PASS',checks:['full_table_and_pagination','truth_column','search_across_stages','budget_flags','selected_candidate','numeric_sort','full_CSV_download','round_navigation','exact_LLM_prompts_and_response','native_KF_details','ordered_log'],network_calls:0,browser_processes:0}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
