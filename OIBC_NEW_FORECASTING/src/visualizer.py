"""Self-contained offline canvas dashboard; no CDN or network operations."""
import json
from pathlib import Path


def render(path,frame,comparison):
    records=[]
    for row in frame.itertuples():
        records.append({'model':row.model,'candidate':row.candidate,'split':row.split,'group':row.group,
                        'ts':row.ts.isoformat(),'actual':float(row.energy_kwh),'pred':float(row.forecast_kwh),
                        'hour':row.ts.hour,'month':row.ts.month})
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>SYNTHETIC forecast diagnostics</title>
<style>body{font:16px system-ui;margin:24px}canvas{width:100%;height:280px;border:1px solid #bbb}section{margin:20px 0}select,input,button{padding:8px;margin:4px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #aaa}.scroll{overflow:auto}</style>
<h1>SYNTHETIC forecasting diagnostics</h1><p>Not competition evidence. KST dates; fixed historical holdout and validation results.</p>
<label>Model <select id="model"></select></label><label>Grouping <select id="candidate"></select></label><label>Split <select id="split"></select></label><label>Group <select id="group"><option>G1</option><option>G2</option></select></label>
<label>From <input id="from" type="date"></label><label>Through <input id="through" type="date"></label>
<button id="png">Export forecast chart PNG</button><button id="html">Export this HTML</button>
<p id="summary"></p><section><h2>Actual (blue) versus predicted (orange), kWh</h2><canvas id="series"></canvas></section>
<section><h2>Signed forecast error over time, kWh</h2><canvas id="error"></canvas></section>
<section><h2>MAE by hour, kWh</h2><canvas id="hour"></canvas></section><section><h2>MAE by month, kWh</h2><canvas id="month"></canvas></section>
<section><h2>Model / grouping / G1 versus G2 comparison (full saved folds)</h2><div class="scroll" id="table"></div></section>
<p>Plant-level error distributions and day/night statistics are preserved in plant_errors_*.csv and error_statistics_*.csv. This view charts aggregated errors only.</p>
<script>const D=__DATA__,C=__COMPARISON__;
for(const id of ['model','candidate','split']){const e=document.getElementById(id);[...new Set(D.map(r=>r[id]))].forEach(v=>{const o=document.createElement('option');o.textContent=v;e.append(o)})}
const dates=D.map(r=>r.ts.slice(0,10)).sort();document.getElementById('from').value=dates[0];document.getElementById('through').value=dates[dates.length-1];
function chart(id,series,labels){const e=document.getElementById(id);e.width=1100;e.height=300;const x=e.getContext('2d');x.fillStyle='white';x.fillRect(0,0,e.width,e.height);const vals=[0,...series.flat()],lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1;x.fillStyle='#222';x.font='12px sans-serif';x.fillText('max '+hi.toFixed(3),10,15);x.fillText('min '+lo.toFixed(3),10,285);x.strokeStyle='#999';let zero=260-(0-lo)/span*220;x.beginPath();x.moveTo(60,zero);x.lineTo(1070,zero);x.stroke();series.forEach((s,k)=>{x.strokeStyle=['#2563eb','#d97706'][k%2];x.beginPath();s.forEach((v,i)=>{const px=60+i*1000/Math.max(1,s.length-1),py=260-(v-lo)/span*220;if(i)x.lineTo(px,py);else x.moveTo(px,py)});x.stroke()});x.fillStyle='#222';x.fillText(labels[0]||'',60,295);x.fillText(labels[labels.length-1]||'',850,295)}
function render(){const f=D.filter(r=>['model','candidate','split','group'].every(k=>r[k]===document.getElementById(k).value)&&r.ts.slice(0,10)>=document.getElementById('from').value&&r.ts.slice(0,10)<=document.getElementById('through').value).sort((a,b)=>a.ts.localeCompare(b.ts));document.getElementById('summary').textContent=f.length+' group-hours selected';chart('series',[f.map(r=>r.actual),f.map(r=>r.pred)],f.map(r=>r.ts));chart('error',[f.map(r=>r.pred-r.actual)],f.map(r=>r.ts));for(const key of ['hour','month']){const by={};f.forEach(r=>{(by[r[key]]??=[]).push(Math.abs(r.pred-r.actual))});const keys=Object.keys(by).sort((a,b)=>a-b);chart(key,[keys.map(k=>by[k].reduce((a,b)=>a+b,0)/by[k].length)],keys)}}
for(const id of ['model','candidate','split','group','from','through'])document.getElementById(id).addEventListener('change',render);
const table=document.createElement('table'),keys=['model','candidate','split','group','mae','rmse','nmae_capacity'];const head=document.createElement('tr');keys.forEach(k=>{const th=document.createElement('th');th.textContent=k;head.append(th)});table.append(head);C.forEach(r=>{const tr=document.createElement('tr');keys.forEach(k=>{const td=document.createElement('td');td.textContent=typeof r[k]==='number'?r[k].toPrecision(6):r[k];tr.append(td)});table.append(tr)});document.getElementById('table').append(table);
function download(url,name){const a=document.createElement('a');a.href=url;a.download=name;a.click()}
document.getElementById('png').onclick=()=>download(document.getElementById('series').toDataURL('image/png'),'synthetic_forecast.png');document.getElementById('html').onclick=()=>{const url=URL.createObjectURL(new Blob(['<!doctype html>'+document.documentElement.outerHTML],{type:'text/html'}));download(url,'synthetic_diagnostics.html');setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></html>'''
    html=html.replace('__DATA__',json.dumps(records,allow_nan=False).replace('<','\\u003c'))
    html=html.replace('__COMPARISON__',comparison.to_json(orient='records').replace('<','\\u003c'))
    Path(path).write_text(html,encoding='utf-8')
