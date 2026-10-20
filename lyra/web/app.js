const $=id=>document.getElementById(id);
let initialized=false,points=[],lastState=null,polling=false,retry=1200,toastTimer;
const names={running:'Tanítás folyamatban',starting:'Indulás',complete:'Befejezve',cancelled:'Elmentve, leállítva',interrupted:'Megszakadt · folytatható',failed:'Hiba · részletek a naplóban',checked:'Ellenőrzés kész'};
function toast(message,error=false){$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').style.display='none',8000)}
async function api(path,body){const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(15000)});const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail));return value}
async function action(path,body={}){try{const r=await api(path,body);toast(r.message||'Elindult.');await poll()}catch(e){toast(e.message,true)}}
document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>{document.querySelectorAll('.tab,.page').forEach(x=>x.classList.remove('active'));button.classList.add('active');$(button.dataset.tab).classList.add('active');draw()});
$('trainButton').onclick=()=>action('/api/train',{run:$('runName').value.trim(),config:{cycles:Number($('cycles').value),batch_size:Number($('batch').value),workers:Number($('workers').value)}});
$('stopButton').onclick=()=>action('/api/stop');
$('benchmarkButton').onclick=()=>action('/api/benchmark',{batch_size:Number($('batch').value)});
$('evaluateButton').onclick=()=>action('/api/evaluate',{run:$('runName').value.trim()});
$('prepareButton').onclick=()=>action('/api/prepare',{source:$('source').value.trim()});
$('runName').onchange=()=>poll();
$('musicFile').onchange=async()=>{const file=$('musicFile').files[0];if(!file)return;try{const form=new FormData();form.append('file',file);const response=await fetch('/api/upload',{method:'POST',body:form});const r=await response.json();if(!response.ok)throw Error(r.detail);$('musicPath').value=r.path;toast('A zene kiválasztva.')}catch(e){toast(e.message,true)}};
$('generateButton').onclick=()=>{const optional=id=>$(id).value.trim()===''?null:Number($(id).value);action('/api/generate',{checkpoint:$('checkpointInput').value.trim(),inputs:$('musicPath').value.trim(),output:$('output').value.trim(),config:{stars:Number($('stars').value),preset:$('preset').value,candidates:Number($('candidates').value),bpm:optional('bpm'),offset:optional('offset'),timing_map:$('timingMap').value.trim()||null}})};
function numeric(v,digits=2){return Number.isFinite(v)?v.toLocaleString('hu-HU',{maximumFractionDigits:digits,minimumFractionDigits:digits}):'—'}
function duration(seconds){if(!Number.isFinite(seconds))return '—';if(seconds>=86400)return numeric(seconds/86400,1)+' nap';if(seconds>=3600)return numeric(seconds/3600,1)+' óra';return Math.max(1,Math.round(seconds/60))+' perc'}
function render(s){lastState=s;const cfg=s.settings;
 if(!initialized){for(const [id,key] of [['runName','run'],['cycles','cycles'],['batch','batch_size'],['workers','workers'],['source','source'],['stars','stars'],['preset','preset'],['candidates','candidates'],['output','output']])$(id).value=cfg[key];$('checkpointInput').value=s.best_checkpoint;initialized=true}
 const r=s.run||{},active=s.job&&['running','starting'].includes(s.job.status);
 $('connection').className='connection';$('connection').textContent='Kapcsolódva';
 const selectedTrain=s.job?.action==='train'&&s.job.arguments?.run_dir===s.run_path;
 const displayStatus=selectedTrain&&['interrupted','failed'].includes(s.job.status)?s.job.status:r.status;
 $('runStatus').textContent=selectedTrain&&active&&r.status!=='running'?'Folytatás betöltése…':names[displayStatus]||'Még nem indult';$('loss').textContent=numeric(r.loss_ema??r.loss,4);$('step').textContent=r.step===undefined?'—':r.step.toLocaleString('hu-HU');$('speed').textContent=Number.isFinite(r.seconds_per_update)?numeric(r.seconds_per_update)+' s':'—';
 $('progress').value=r.target_steps?Math.min(100,100*r.step/r.target_steps):0;
 $('coverage').textContent=r.step!==undefined?`${(r.visited_windows??0).toLocaleString('hu-HU')} ablak · ${numeric((r.completed_cycles||0)+(r.cycle_fraction||0),2)} teljes adatkör · cél ${(r.target_steps||0).toLocaleString('hu-HU')} update`:s.dataset.complete?'A teljes train készlet elkészült. A tanítás indítható.':'A teljes dataset előkészítése után indítható.';
 $('checkpoint').textContent=s.last_checkpoint?'Utolsó mentés: '+s.last_checkpoint:s.run_path;
 $('timeInfo').textContent=r.step?`Eltelt: ${duration(r.elapsed_seconds)}${r.status==='running'&&r.step>=10&&s.speed_average?' · Becsült hátralévő idő: '+duration(Math.max(0,(r.target_steps-r.step)*s.speed_average)):''}`:'';
 const run=encodeURIComponent($('runName').value.trim());$('csvExport').href='/api/metrics/export?run='+run+'&format=csv';$('jsonlExport').href='/api/metrics/export?run='+run+'&format=jsonl&include_archived=true';
 $('trainButton').textContent=s.last_checkpoint?'Tanítás folytatása':'Tanítás indítása';
 for(const id of ['trainButton','generateButton','prepareButton','benchmarkButton','evaluateButton'])$(id).disabled=!!active;
 $('stopButton').disabled=!active;
 $('log').textContent=s.job?.log||s.job?.message||'Még nincs futó feladat.';$('jobLabel').textContent=s.job?`${s.job.action} · ${s.job.status}`:'';
 if(selectedTrain&&s.job?.error)$('runStatus').textContent=s.job.error;
 const counts=s.dataset.maps||{};$('dataStats').replaceChildren();for(const [key,label] of [['train','Tanítómap'],['validation','Validáció'],['test','Teszt']]){const div=document.createElement('div'),span=document.createElement('span'),strong=document.createElement('strong');span.textContent=label;strong.textContent=(counts[key]||0).toLocaleString('hu-HU');div.append(span,strong);$('dataStats').append(div)}
 $('prepareProgress').textContent=s.dataset.complete?`Előkészítve: ${Object.values(s.dataset.windows).reduce((a,b)=>a+b,0).toLocaleString('hu-HU')} ablak. Kizárt mapok: ${s.dataset.rejected}.`:s.preparation.maps?`${s.preparation.maps.toLocaleString('hu-HU')} / ${s.preparation.total.toLocaleString('hu-HU')} map feldolgozva.`:'A meglevő teljes adathalmaz előkészítése szükséges.';
 $('dataPath').textContent=s.dataset.index||'';
 if(s.gpu_check.passed)$('gpuCheck').textContent=s.gpu_check.warning||`Sikeres próba · batch ${s.gpu_check.batch_size||1} · ${numeric(s.gpu_check.parameters/1e6,1)}M paraméter · ${numeric(s.gpu_check.peak_vram_gib)} GiB csúcs · ${numeric(s.gpu_check.seconds_per_microbatch)} s / maximális microbatch`;
 if(!$('checkpointInput').value&&s.best_checkpoint)$('checkpointInput').value=s.best_checkpoint;
 $('outputs').replaceChildren();const result=s.job?.action==='generate'&&Array.isArray(s.job.result)?s.job.result:[];for(const path of result){const a=document.createElement('a');a.href='/api/download?path='+encodeURIComponent(path);a.textContent=path.split(/[\\/]/).pop();$('outputs').append(a)}
 points=s.history||[];$('emptyChart').style.display=points.length?'none':'grid';draw();
}
function draw(){const canvas=$('lossChart'),rect=canvas.getBoundingClientRect();if(rect.width<1)return;const dpr=devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const W=rect.width,H=rect.height,L=38,R=12,T=12,B=25;c.clearRect(0,0,W,H);const values=points.flatMap(p=>[p.loss,p.validation_token_loss]).filter(Number.isFinite);const sorted=[...values].sort((a,b)=>a-b),ymax=Math.max(.1,(sorted[Math.floor(sorted.length*.98)]||1)*1.15),xmin=points[0]?.step||0,xmax=Math.max(xmin+1,points.at(-1)?.step||1);c.font='10px Segoe UI';c.lineWidth=1;c.strokeStyle='#27374a';c.fillStyle='#7e93ab';for(let i=0;i<5;i++){const y=T+(H-T-B)*i/4;c.beginPath();c.moveTo(L,y);c.lineTo(W-R,y);c.stroke();c.fillText((ymax*(1-i/4)).toFixed(1),3,y+3)}for(const [key,color] of [['loss','#70e0c3'],['validation_token_loss','#b5a1f7']]){c.strokeStyle=color;c.lineWidth=1.7;c.beginPath();let started=false;for(const p of points){if(!Number.isFinite(p[key]))continue;const x=L+(p.step-xmin)/(xmax-xmin)*(W-L-R),y=T+(1-Math.min(ymax,p[key])/ymax)*(H-T-B);if(!started){c.moveTo(x,y);started=true}else c.lineTo(x,y)}c.stroke()}c.fillText(String(xmin),L,H-5);c.textAlign='right';c.fillText(xmax.toLocaleString('hu-HU')+' step',W-R,H-5)}
async function poll(){if(polling)return;polling=true;try{render(await api('/api/state'+(initialized?'?run='+encodeURIComponent($('runName').value.trim()):'')));retry=1500}catch(e){$('connection').className='connection offline';$('connection').textContent='Újracsatlakozás… a mentett előzmény megmarad';retry=Math.min(15000,Math.max(2000,retry*1.5))}finally{polling=false}}
async function loop(){await poll();setTimeout(loop,retry)}
window.addEventListener('resize',draw);document.addEventListener('visibilitychange',()=>{if(!document.hidden)poll()});loop();
