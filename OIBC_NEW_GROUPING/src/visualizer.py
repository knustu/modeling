"""외부 자원 없이 동작하는 로컬 HTML 시각화."""
import json
from pathlib import Path
import numpy as np


def write_visualization(path,sites,p,z,candidates,logs,rows):
    u,s,_ = np.linalg.svd(z-z.mean(axis=0),full_matrices=False)
    xy = np.zeros((len(sites),2))
    d = min(2,len(s))
    xy[:,:d] = u[:,:d]*s[:d]
    c = sites.capacity_kw.to_numpy(float)
    complete = p.dropna(how='any')
    profiles = {}
    for name,x in candidates.items():
        profiles[name] = []
        for mask in (x,~x):
            v = complete.loc[:,mask].sum(axis=1)/c[mask].sum()
            hourly = v.groupby(v.index.hour).mean()
            profiles[name].append([float(hourly[h]) if h in hourly.index else None for h in range(24)])
    data = {'ids':sites.site_id.tolist(),'capacity':c.tolist(),'xy':xy.tolist(),
            'groups':{k:v.astype(int).tolist() for k,v in candidates.items()},
            'profiles':profiles,'logs':logs,'metrics':[{k:r[k] for k in ('candidate','g1_count','g2_count','g1_capacity_kw','g2_capacity_kw','capacity_imbalance_kw','objective','runtime_seconds')} for r in rows]}
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>독립 그룹 편성 진단</title>
<style>body{font:16px system-ui;margin:24px;color:#17212b}section{border:1px solid #ccd3dc;padding:16px;margin:12px 0;min-width:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}svg{width:100%;height:auto}td,th{border:1px solid #ccd3dc;padding:8px}table{border-collapse:collapse}.scroll{overflow:auto}select,button{padding:8px}</style>
<h1>Phase 1 그룹 편성 진단</h1><p>과거 대리 지표입니다. 예측오차 또는 수익 개선을 입증하지 않습니다.</p>
<label>후보 <select id="candidate"></select></label><button id="export">산점도 SVG 이미지 저장</button>
<p id="inspect">발전소 점에 마우스를 올리거나 클릭하세요. G1은 파랑, G2는 주황입니다.</p>
<div class="grid"><section><h2>과거 특징 투영</h2><div id="scatter"></div><p>좌표가 아닌 SVD 특징 축입니다.</p></section><section><h2>설비용량 비교</h2><div id="capacity"></div></section><section><h2>시간대별 과거 발전량</h2><div id="profile"></div><p>전체 발전소 관측 완료 시간만 사용합니다. 단위: kWh/kW.</p></section><section><h2>탐색 수렴</h2><div id="trace"></div><p>용량 후보·프로파일 후보·최적화 후보의 수렴 목적함수는 서로 다릅니다.</p></section></div>
<section><h2>후보 나란히 비교</h2><p>비교표 목적함수는 공통 설정 가중치로 다시 계산합니다.</p><div class="scroll" id="comparison"></div></section>
<script>
const D=__DATA__,ns='http://www.w3.org/2000/svg',colors=['#2563eb','#d97706'];
function el(tag,attrs={},value){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(value!==undefined)e.textContent=value;return e}
function svg(id){const e=el('svg',{viewBox:'0 0 560 340',role:'img'});document.getElementById(id).replaceChildren(e);return e}
function text(s,x,y,t){s.append(el('text',{x,y,'font-size':12},t))}
function scale(values,a,b){let vals=values.filter(Number.isFinite),lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1;if(lo===hi){lo-=1;hi+=1}return v=>a+(v-lo)*(b-a)/(hi-lo)}
function chart(id,series,xlabel,ylabel){const s=svg(id),vals=series.flat().filter(Number.isFinite);if(!vals.length){text(s,30,100,'완전 관측 자료가 부족합니다.');return}const sy=scale([0,...vals],285,35),n=Math.max(...series.map(v=>v.length));series.forEach((values,g)=>{let prev=null;values.forEach((v,i)=>{if(v===null){prev=null;return}const pt=[45+i*480/Math.max(1,n-1),sy(v)];if(prev)s.append(el('line',{x1:prev[0],y1:prev[1],x2:pt[0],y2:pt[1],stroke:colors[g%2],'stroke-width':2}));const dot=el('circle',{cx:pt[0],cy:pt[1],r:3,fill:colors[g%2]});dot.append(el('title',{},i+': '+v));s.append(dot);prev=pt})});text(s,35,320,xlabel);text(s,15,20,ylabel);text(s,400,20,'최대 '+Math.max(...vals).toPrecision(5))}
function scatter(id,name){const s=svg(id),groups=D.groups[name],sx=scale(D.xy.map(v=>v[0]),40,520),sy=scale(D.xy.map(v=>v[1]),285,35);D.xy.forEach((v,i)=>{const label=D.ids[i]+' | '+D.capacity[i]+' kW | '+(groups[i]?'G1':'G2'),dot=el('circle',{cx:sx(v[0]),cy:sy(v[1]),r:4,fill:colors[groups[i]?0:1],tabindex:0});dot.append(el('title',{},label));['mouseover','click','focus'].forEach(event=>dot.addEventListener(event,()=>document.getElementById('inspect').textContent=label));s.append(dot)});text(s,35,320,'SVD 특징 1');text(s,15,20,'SVD 특징 2')}
function render(){const name=document.getElementById('candidate').value;scatter('scatter',name);const caps=[0,0];D.groups[name].forEach((g,i)=>caps[g?0:1]+=D.capacity[i]);const s=svg('capacity'),mx=Math.max(...caps);caps.forEach((v,i)=>{s.append(el('rect',{x:60,y:50+i*110,width:420*v/mx,height:45,fill:colors[i]}));text(s,15,80+i*110,'G'+(i+1));text(s,60,120+i*110,v.toFixed(3)+' kW')});chart('profile',D.profiles[name],'KST 구간 시작 시각 0–23','설비용량 정규화 발전량');chart('trace',[D.logs[name].map(v=>v.objective)],'승인된 단계','해당 탐색 목적함수')}
const select=document.getElementById('candidate');Object.keys(D.groups).forEach(name=>{const o=document.createElement('option');o.textContent=name;select.append(o)});select.addEventListener('change',render);
const table=document.createElement('table'),keys=['candidate','g1_count','g2_count','g1_capacity_kw','g2_capacity_kw','capacity_imbalance_kw','objective','runtime_seconds'],labels=['후보','G1 발전소 수','G2 발전소 수','G1 용량(kW)','G2 용량(kW)','용량 차(kW)','공통 목적함수','시간(초)'],head=document.createElement('tr');labels.forEach(t=>{const th=document.createElement('th');th.textContent=t;head.append(th)});table.append(head);D.metrics.forEach(r=>{const tr=document.createElement('tr');keys.forEach(k=>{const td=document.createElement('td');td.textContent=typeof r[k]==='number'?r[k].toPrecision(7):r[k];tr.append(td)});table.append(tr)});document.getElementById('comparison').append(table);
Object.keys(D.groups).forEach((name,i)=>{const section=document.createElement('section'),h=document.createElement('h3'),div=document.createElement('div');h.textContent=name;div.id='side_'+i;section.append(h,div);document.getElementById('comparison').append(section);scatter(div.id,name)});
document.getElementById('export').addEventListener('click',()=>{const image=document.querySelector('#scatter svg').cloneNode(true);image.setAttribute('xmlns',ns);const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(image)],{type:'image/svg+xml'})),a=document.createElement('a');a.href=url;a.download='grouping_scatter.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});render();
</script></html>'''
    encoded = json.dumps(data,ensure_ascii=False,allow_nan=False,default=lambda v:v.item()).replace('<','\\u003c')
    Path(path).write_text(html.replace('__DATA__',encoded),encoding='utf-8')
