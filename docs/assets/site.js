'use strict';
const $ = (selector) => document.querySelector(selector);
const P = {ref:'var(--chart-ref)', ml:'var(--chart-ml)', grid:'var(--chart-grid)', text:'var(--muted)'};
let DATA = null;
let POINTS_BY_MOLECULE = new Map();
let MOL_BY_ID = new Map();
let currentMolecule = null;
const format = (n,d=2) => typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}) : 'n/a';
const pc = n => `${format(n,2)}%`;
function addText(node, text){node.textContent=text;return node;}
function toNode(tag, classes, text){const node=document.createElement(tag);if(classes)node.className=classes;if(text!==undefined)addText(node,text);return node;}
function makeSVG(tag,attrs={},text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));if(text!==undefined)el.textContent=text;return el;}
function stats(){
  const s=DATA.thermo_summary;
  addText($('#stat-molecules'), String(DATA.metadata.test_molecules));
  addText($('#stat-pressure'),pc(s.mean_pressure_ape_percent_on_comparable));
  addText($('#stat-density'),pc(s.mean_density_ape_percent_on_comparable));
  addText($('#stat-success'),`${s.n_comparable_pressure}/${s.n_attempted}`);
  const stamp=new Date(DATA.metadata.generated_utc);
  addText($('#footer-generated'),`Exported ${Number.isNaN(stamp.getTime())?'from frozen CSVs':stamp.toLocaleDateString('en-IN',{year:'numeric',month:'short',day:'numeric'})}`);
}
function drawChart(root, rows, category){
  root.replaceChildren();
  if(!rows || rows.length===0){addText(root,'No matching values available.');return;}
  const W=520,H=315, left=63,right=20,top=23,bottom=52;
  const pkey=category==='pressure'?'pressure_reference_bar':'density_reference_kg_m3';
  const mkey=category==='pressure'?'pressure_ml_bar':'density_ml_kg_m3';
  const vals=rows.flatMap(r=>[r[pkey],r[mkey]]);
  const minY=Math.min(...vals),maxY=Math.max(...vals);
  const pad=Math.max((maxY-minY)*.18, maxY*.018);
  const y0=Math.max(0,minY-pad),y1=maxY+pad;
  const temps=rows.map(r=>r.temperature_K),t0=Math.min(...temps),t1=Math.max(...temps);
  const fx=t=>left+(t-t0)/(t1-t0||1)*(W-left-right);
  const fy=y=>H-bottom-(y-y0)/(y1-y0||1)*(H-top-bottom);
  const svg=makeSVG('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':category==='pressure'?'Pressure versus temperature':'Liquid density versus temperature'});
  svg.append(makeSVG('rect',{x:0,y:0,width:W,height:H,fill:'var(--chart-bg)'}));
  for(let i=0;i<=4;i++){
    const v=y0+(y1-y0)*i/4;
    const y=fy(v);
    svg.append(makeSVG('line',{x1:left,x2:W-right,y1:y,y2:y,stroke:P.grid,'stroke-width':1}));
    svg.append(makeSVG('text',{x:left-10,y:y+4,'text-anchor':'end'},format(v,category==='pressure'?1:0)));
  }
  rows.forEach(row=>{
    const x=fx(row.temperature_K);
    svg.append(makeSVG('line',{x1:x,x2:x,y1:top,y2:H-bottom,stroke:P.grid,'stroke-dasharray':'2 7'}));
    svg.append(makeSVG('text',{x:x,y:H-bottom+19,'text-anchor':'middle'},format(row.temperature_K,0)));
  });
  svg.append(makeSVG('line',{x1:left,x2:W-right,y1:H-bottom,y2:H-bottom,stroke:'var(--chart-grid)'}));
  svg.append(makeSVG('text',{x:(W+left-right)/2,y:H-10,'text-anchor':'middle',class:'axis-title'},'Temperature (K)'));
  [[pkey,P.ref,'Published parameters','ref-line'],[mkey,P.ml,'ML parameters','ml-line']].forEach(([key,color,label,cls])=>{
    const d=rows.map((row,i)=>`${i?'L':'M'} ${fx(row.temperature_K).toFixed(2)} ${fy(row[key]).toFixed(2)}`).join(' ');
    svg.append(makeSVG('path',{d,class:`plot-line ${cls}`}));
    rows.forEach(row=>{
      const c=makeSVG('circle',{cx:fx(row.temperature_K),cy:fy(row[key]),r:5,fill:color,class:'data-point',tabindex:0});
      c.append(makeSVG('title',{},`${label}\nT = ${format(row.temperature_K,2)} K\n${category==='pressure'?'P = '+format(row[key],4)+' bar':'ρ = '+format(row[key],2)+' kg/m³'}`));
      svg.append(c);
    });
  });
  root.append(svg);
}
function selectMolecule(smiles, scroll=false){
  const mol=MOL_BY_ID.get(smiles);
  if(!mol)return;
  currentMolecule=smiles;
  $('#molecule-select').value=smiles;
  addText($('#selected-name'),mol.name);
  addText($('#selected-smiles'),`SMILES: ${smiles}`);
  addText($('#selected-pressure-error'),pc(mol.mean_pressure_ape_percent));
  addText($('#selected-density-error'),pc(mol.mean_density_ape_percent));
  const rows=POINTS_BY_MOLECULE.get(smiles).slice().sort((a,b)=>a.temperature_K-b.temperature_K);
  drawChart($('#pressure-chart'),rows,'pressure');
  drawChart($('#density-chart'),rows,'density');
  const first=rows[0];
  const params=[['Segment number m','m',''],['Diameter σ','sigma','Å'],['Dispersion energy ε/k','epsilon_k','K']];
  const tbody=$('#parameter-rows');tbody.replaceChildren();
  for(const [label,key,unit] of params){
    const tr=document.createElement('tr');
    [label,`${format(first['reference_'+key],4)} ${unit}`.trim(),`${format(first['predicted_'+key],4)} ${unit}`.trim()].forEach(value=>tr.append(toNode('td','',value)));
    tbody.append(tr);
  }
  const statebody=$('#state-rows');statebody.replaceChildren();
  for(const row of rows){
    const tr=document.createElement('tr');
    [format(row.temperature_K,1),format(row.temperature_fraction_of_ref_Tc,2),format(row.pressure_reference_bar,3),format(row.pressure_ml_bar,3),pc(row.pressure_ape_percent)].forEach(x=>tr.append(toNode('td','',x)));
    statebody.append(tr);
  }
  renderMoleculeTable();
  if(scroll)$('#explorer').scrollIntoView({behavior:'smooth',block:'start'});
}
function renderMoleculeTable(){
  const query=$('#table-search').value.trim().toLowerCase();
  const sort=$('#sort-choice').value;
  const list=DATA.per_molecule.filter(m=>`${m.name} ${m.smiles}`.toLowerCase().includes(query));
  if(sort==='pressure-desc')list.sort((a,b)=>b.mean_pressure_ape_percent-a.mean_pressure_ape_percent);
  if(sort==='pressure-asc')list.sort((a,b)=>a.mean_pressure_ape_percent-b.mean_pressure_ape_percent);
  if(sort==='density-desc')list.sort((a,b)=>b.mean_density_ape_percent-a.mean_density_ape_percent);
  if(sort==='name')list.sort((a,b)=>a.name.localeCompare(b.name));
  const tbody=$('#all-molecules-body');tbody.replaceChildren();
  if(!list.length){let tr=toNode('tr');let td=toNode('td','empty-table','No matching molecules.');td.colSpan=5;tr.append(td);tbody.append(tr);return;}
  list.forEach(m=>{
    const tr=toNode('tr');tr.tabIndex=0;tr.setAttribute('aria-label',`Open ${m.name} in explorer`);
    const name=toNode('td','molecule-name',m.name);
    const smiles=toNode('td','smiles-cell',m.smiles);smiles.title=m.smiles;
    const pe=toNode('td','error-value'+(m.mean_pressure_ape_percent>=35?' high':''),pc(m.mean_pressure_ape_percent));
    const de=toNode('td','',pc(m.mean_density_ape_percent));
    const counts=toNode('td','',`${m.n_comparable_pressure}/${m.n_attempted}`);
    tr.append(name,smiles,pe,de,counts);
    tr.addEventListener('click',()=>selectMolecule(m.smiles,true));
    tr.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectMolecule(m.smiles,true);}});
    if(m.smiles===currentMolecule)tr.setAttribute('aria-current','true');
    tbody.append(tr);
  });
}
function development(){
  const rows=DATA.development.development_vle||[];
  const root=$('#development-bars');root.replaceChildren();
  if(!rows.length){root.append(toNode('p','table-note','Development results not exported. Final-test results above are still available.'));return;}
  const vals=rows.map(r=>({name:r.model.replaceAll('_',' + ').replace('ExtraTrees + 156','Extra Trees · 156').replace('Ridge + 5','Ridge · 5').replace('Ridge + 12','Ridge · 12'),value:r.macro_mean_molecule_ape_percent,model:r.model})).filter(x=>Number.isFinite(x.value)).sort((a,b)=>a.value-b.value);
  const max=Math.max(1,...vals.map(x=>x.value));
  vals.forEach((r,i)=>{
    const row=toNode('div','bar-row');row.append(toNode('div','bar-label',r.name));
    const track=toNode('div','bar-track');const fill=toNode('div','bar-fill'+(i===0?' best':''));fill.style.width=`${100*r.value/max}%`;track.append(fill);row.append(track,toNode('div','bar-number',pc(r.value)));root.append(row);
  });
  const drows=DATA.development.development_density||[];
  const text=drows.length?drows.map(r=>`${r.model.replace('_',' ')}: ${pc(r.mean_density_ape_percent)}`).join(' · '):'Density development comparison not included in export.';
  addText($('#development-density'),`Development saturated-liquid density: ${text}`);
}
function sensitivity(){
  const root=$('#sensitivity-content');root.replaceChildren();
  const rows=(DATA.development.sensitivities||[]).filter(r=>r.status==='ok'&&Number.isFinite(r.S_dlnP_dlnparameter));
  if(!rows.length){root.append(toNode('p','table-note','Sensitivity CSV not included in export. The qualitative findings are reported separately.'));return;}
  const table=toNode('table','sensitivity-table');
  const head=toNode('thead');const hr=toNode('tr');['Molecule','m','σ','ε/k'].forEach(x=>hr.append(toNode('th','',x)));head.append(hr);table.append(head);
  const body=toNode('tbody');
  const names=[...new Set(rows.map(r=>r.name))];
  names.forEach(name=>{
    const group=rows.filter(r=>r.name===name);const val=p=>{const item=group.find(x=>x.parameter===p);return item?format(item.S_dlnP_dlnparameter,2):'—'};
    const tr=toNode('tr');[name,val('m'),val('sigma'),val('epsilon_k')].forEach(t=>tr.append(toNode('td','',t)));body.append(tr);
  });
  table.append(body);root.append(table);
  root.append(toNode('p','table-note','Local signed sensitivities at one reference-defined temperature per molecule; not full uncertainty propagation.'));
}
function downloadCSV(){
  if(!currentMolecule)return;
  const rows=POINTS_BY_MOLECULE.get(currentMolecule);
  const fields=['name','smiles','temperature_K','temperature_fraction_of_ref_Tc','pressure_reference_bar','pressure_ml_bar','pressure_ape_percent','density_reference_kg_m3','density_ml_kg_m3','density_ape_percent'];
  const csv=[fields.join(','),...rows.map(row=>fields.map(key=>'"'+String(row[key]??'').replaceAll('"','""')+'"').join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download='pcsaft_heldout_'+currentMolecule.replace(/[^a-z0-9]/gi,'_').slice(0,45)+'.csv';document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
// Research experiment dashboard, populated only from exported source CSV data.
const LAB_FEATURES = [
  {key:'Original_5', n:5, label:'Original · 5', short:'5'},
  {key:'Enhanced_12', n:12, label:'Enhanced · 12', short:'12'},
  {key:'Richer_28', n:28, label:'Richer · 28', short:'28'},
  {key:'Richer_28_Morgan128', n:156, label:'Morgan + 28 · 156', short:'156'}
];
const LAB_MODELS = [
  {key:'Ridge', label:'Ridge', color:'var(--model-ridge)'},
  {key:'Random_Forest', label:'Random Forest', color:'var(--model-rf)'},
  {key:'Extra_Trees', label:'Extra Trees', color:'var(--model-extra)'}
];
const DEV_MODELS = [
  {key:'Ridge_5', label:'Ridge · 5', color:'var(--model-ridge)'},
  {key:'Ridge_12', label:'Ridge · 12', color:'var(--model-rf)'},
  {key:'ExtraTrees_156', label:'Extra Trees · 156', color:'var(--model-extra)'}
];
const TARGET_LABELS = {
  m:{label:'Segment number m',unit:'(dimensionless)'},
  sigma:{label:'Segment diameter σ',unit:'Å'},
  epsilon_k:{label:'Dispersion energy ε/k',unit:'K'}
};
const LAB = {target:'epsilon_k',metric:'mae',model:'all',feature:null,property:'pressure'};
function labScore(row){return row?.[`cv_${LAB.metric}_mean`];}
function labStd(row){return row?.[`cv_${LAB.metric}_std`];}
function labRows(){return DATA.experiments.cv_models.filter(r=>r.target===LAB.target);}
function labFind(feature,model){return labRows().find(r=>r.features===feature&&r.model===model);}
function labModel(key){return LAB_MODELS.find(m=>m.key===key);}
function labUnit(){return TARGET_LABELS[LAB.target].unit;}
function labNumber(value){return format(value,LAB.target==='epsilon_k'?2:3);}
function addSvgText(svg,x,y,text,attrs={}){svg.append(makeSVG('text',{x,y,...attrs},text));}
function labDrawChart(){
  const root=$('#cv-evolution-chart');root.replaceChildren();
  const rows=labRows().filter(r=>Number.isFinite(labScore(r)));
  if(!rows.length){root.append(toNode('p','table-note','No comparison metrics found.'));return;}
  const metricLabel=LAB.metric==='mae'?'MAE':'RMSE';
  addText($('#cv-chart-title'),`${TARGET_LABELS[LAB.target].label} · ${metricLabel}`);
  const W=750,H=350,margin={l:76,r:20,t:26,b:68};
  const xl=[margin.l,margin.l+(W-margin.l-margin.r)/3,margin.l+2*(W-margin.l-margin.r)/3,W-margin.r];
  const maxVal=Math.max(...rows.map(r=>labScore(r)+Math.max(0,labStd(r)||0)))*1.12;
  const yr=v=>H-margin.b-v/Math.max(.001,maxVal)*(H-margin.b-margin.t);
  const svg=makeSVG('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`${metricLabel} across feature counts for ${TARGET_LABELS[LAB.target].label}; colors identify algorithms`});
  svg.append(makeSVG('rect',{width:W,height:H,fill:'transparent'}));
  for(let i=0;i<=4;i++){
    const v=maxVal*i/4,y=yr(v);
    svg.append(makeSVG('line',{x1:margin.l,x2:W-margin.r,y1:y,y2:y,stroke:'var(--chart-grid)'}));
    addSvgText(svg,margin.l-12,y+4,LAB.target==='epsilon_k'?format(v,1):format(v,2),{'text-anchor':'end',class:'lab-axis'});
  }
  LAB_FEATURES.forEach((level,i)=>{
    const x=xl[i];
    if(LAB.feature===level.key){svg.append(makeSVG('rect',{x:x-39,y:margin.t,width:78,height:H-margin.b-margin.t,rx:6,fill:'var(--chart-highlight)',opacity:.8}));}
    svg.append(makeSVG('line',{x1:x,x2:x,y1:margin.t,y2:H-margin.b,stroke:'var(--chart-grid)','stroke-dasharray':'3 7'}));
    addSvgText(svg,x,H-margin.b+21,level.short,{'text-anchor':'middle',class:'lab-axis-label'});
    addSvgText(svg,x,H-margin.b+39,level.n===156?'28 + Morgan':'descriptors',{'text-anchor':'middle',class:'lab-tick-caption'});
  });
  LAB_MODELS.forEach(model=>{
    const sub=LAB_FEATURES.map((f,i)=>({row:labFind(f.key,model.key),i})).filter(x=>x.row&&Number.isFinite(labScore(x.row)));
    if(!sub.length)return;
    const isDim=LAB.model!=='all'&&LAB.model!==model.key;
    const transparency=isDim?.20:1;
    const path=sub.map((r,i)=>`${i?'L':'M'} ${xl[r.i].toFixed(2)} ${yr(labScore(r.row)).toFixed(2)}`).join(' ');
    svg.append(makeSVG('path',{d:path,fill:'none',stroke:model.color,'stroke-width':isDim?2:3.6,'stroke-linecap':'round','stroke-linejoin':'round',opacity:transparency}));
    sub.forEach(({row,i})=>{
      const x=xl[i],y=yr(labScore(row)),s=labStd(row)||0;
      const g=makeSVG('g',{opacity:transparency});
      if(s>0){
        g.append(makeSVG('line',{x1:x,x2:x,y1:yr(labScore(row)-s),y2:yr(labScore(row)+s),stroke:model.color,'stroke-width':1.7}));
        for(const yy of [yr(labScore(row)-s),yr(labScore(row)+s)])g.append(makeSVG('line',{x1:x-5,x2:x+5,y1:yy,y2:yy,stroke:model.color,'stroke-width':1.7}));
      }
      const dot=makeSVG('circle',{cx:x,cy:y,r:LAB.feature===row.features?8:6,fill:model.color,stroke:'var(--panel)','stroke-width':2,tabindex:0,'aria-label':`${model.label}, ${LAB_FEATURES[i].n} descriptors, ${metricLabel}: ${labNumber(labScore(row))} ${labUnit()}`});
      dot.append(makeSVG('title',{},`${model.label} · ${LAB_FEATURES[i].n} features\n${metricLabel}: ${labNumber(labScore(row))} ${labUnit()}\nFold standard deviation: ${labNumber(s)} ${labUnit()}\nClick to focus this feature representation`));
      dot.addEventListener('click',()=>labFocusFeature(row.features));
      dot.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();labFocusFeature(row.features);}});
      g.append(dot);svg.append(g);
    });
  });
  addSvgText(svg,margin.l,(margin.t+H-margin.b)/2,'Validation error',{'text-anchor':'middle',class:'sr-svg-label',transform:`rotate(-90 ${margin.l} ${(margin.t+H-margin.b)/2})`});
  root.append(svg);
  const legend=$('#lab-model-controls');legend.replaceChildren();
  LAB_MODELS.forEach(model=>{
    const active=LAB.model===model.key;
    const btn=toNode('button','model-chip'+(active?' chosen':'')+(LAB.model!=='all'&&!active?' subdued':''));
    btn.type='button';btn.setAttribute('aria-pressed',String(active));
    const swatch=toNode('span','model-chip-swatch');swatch.style.background=model.color;
    btn.append(swatch,document.createTextNode(model.label));
    btn.addEventListener('click',()=>{LAB.model=active?'all':model.key;labRender();});legend.append(btn);
  });
  const sorted=rows.slice().sort((a,b)=>labScore(a)-labScore(b));
  const best=sorted[0];
  addText($('#cv-best-text'),`Lowest observed ${metricLabel}: ${labModel(best.model)?.label||best.model} with ${LAB_FEATURES.find(f=>f.key===best.features)?.n} features = ${labNumber(labScore(best))} ${labUnit()}. This is training-only cross-validation, not final-test accuracy.`);
}
function labFocusFeature(feature){LAB.feature=LAB.feature===feature?null:feature;labRender();}
function labDrawCollisions(){
  const root=$('#collision-bars');root.replaceChildren();
  const values=DATA.experiments.collisions;
  const max=Math.max(...values.map(r=>r.colliding_molecules),1);
  values.forEach(item=>{
    const level=LAB_FEATURES.find(f=>f.key===item.features);
    const selected=LAB.feature===item.features;
    const btn=toNode('button','collision-item'+(selected?' selected':''));
    btn.type='button';btn.setAttribute('aria-pressed',String(selected));
    const meta=toNode('div','collision-meta');meta.append(toNode('strong','',level.label),toNode('span','',`${item.unique_vectors}/155 distinct vectors`));
    const track=toNode('div','collision-track');const fill=toNode('div','collision-fill');fill.style.width=`${100*item.colliding_molecules/max}%`;track.append(fill);
    const val=toNode('span','collision-value',String(item.colliding_molecules));
    btn.append(meta,track,val);btn.addEventListener('click',()=>labFocusFeature(item.features));root.append(btn);
  });
  const from=values.find(x=>x.features==='Original_5')?.colliding_molecules;
  const to=values.find(x=>x.features==='Richer_28_Morgan128')?.colliding_molecules;
  if(Number.isFinite(from)&&Number.isFinite(to)){
    const change=from>0?`, a ${format(100*(from-to)/from,1)}% reduction`:'';
    addText($('#collision-insight'),`From ${from} molecules in collision groups to ${to}${change}. This measures representation distinctiveness, not prediction accuracy.`);
  }
}
function labDrawHeatmap(){
  const root=$('#cv-heatmap-body');root.replaceChildren();
  addText($('#heatmap-title'),`${TARGET_LABELS[LAB.target].label} · ${LAB.metric.toUpperCase()} ${labUnit()}`);
  const scored=labRows().map(r=>labScore(r)).filter(Number.isFinite);
  const min=Math.min(...scored),max=Math.max(...scored);
  LAB_FEATURES.forEach(level=>{
    const tr=toNode('tr',LAB.feature===level.key?'heatmap-selected':'');
    const name=toNode('th','heatmap-row-label',String(level.n));name.scope='row';name.title=level.label;
    tr.append(name);
    LAB_MODELS.forEach(model=>{
      const row=labFind(level.key,model.key);const td=toNode('td');
      if(!row){td.append(toNode('span','heatmap-empty','—'));td.title='Configuration not evaluated';}
      else{
        const val=labScore(row);const quality=(max-val)/Math.max(1e-9,max-min);
        const btn=toNode('button','heatmap-cell',labNumber(val));btn.type='button';btn.style.setProperty('--q',String(quality));
        btn.title=`${level.n} features · ${model.label} · ${labNumber(val)} ${labUnit()} (standard deviation ${labNumber(labStd(row))})`;
        btn.addEventListener('click',()=>{LAB.feature=level.key;LAB.model=model.key;labRender();});td.append(btn);
      }
      tr.append(td);
    });root.append(tr);
  });
}
function labDrawOverfit(){
  const root=$('#overfit-bars');root.replaceChildren();
  const rows=(DATA.experiments.overfitting||[]).filter(r=>r.target===LAB.target);
  if(!rows.length){root.append(toNode('p','table-note','Training/validation diagnostic not found.'));return;}
  const max=Math.max(...rows.flatMap(r=>[r[`train_${LAB.metric}`],r[`validation_${LAB.metric}`]]));
  addText($('#gap-unit'),`${LAB.metric.toUpperCase()} ${labUnit()}`);
  DEV_MODELS.forEach(m=>{
    const r=rows.find(x=>x.model===m.key);if(!r)return;
    const train=r[`train_${LAB.metric}`],val=r[`validation_${LAB.metric}`];
    const row=toNode('div','gap-item');
    row.append(toNode('div','gap-name',m.label));
    [['Train',train,'train'],['Validation',val,'val']].forEach(([name,n,kind])=>{
      const line=toNode('div','gap-line');line.append(toNode('span','',name));
      const track=toNode('div','gap-track');const fill=toNode('div','gap-fill '+kind);fill.style.width=`${100*n/Math.max(1e-9,max)}%`;track.append(fill);line.append(track,toNode('strong','',labNumber(n)));row.append(line);
    });
    row.append(toNode('span','gap-ratio',`Validation / train: ${format(val/train,2)}×`));root.append(row);
  });
}
function labDrawProperty(){
  const root=$('#development-property-bars');root.replaceChildren();
  const isPressure=LAB.property==='pressure';
  const data=isPressure?DATA.development.development_vle:DATA.development.development_density;
  const key=isPressure?'macro_mean_molecule_ape_percent':'mean_density_ape_percent';
  const rows=DEV_MODELS.map(m=>({ ...m,value:data.find(r=>r.model===m.key)?.[key]}));
  if(rows.some(r=>!Number.isFinite(r.value))){root.append(toNode('p','table-note','Development property comparison not available.'));return;}
  const max=Math.max(...rows.map(r=>r.value));
  rows.forEach(r=>{
    const line=toNode('div','outcome-item');const label=toNode('div','outcome-label',r.label);
    const track=toNode('div','outcome-track');const fill=toNode('div','outcome-fill');fill.style.width=`${100*r.value/max}%`;fill.style.background=r.color;track.append(fill);
    const num=toNode('strong','',pc(r.value));line.append(label,track,num);root.append(line);
  });
}
function labDrawHoldout(){
  const root=$('#holdout-baseline-bars');root.replaceChildren();
  const metrics=DATA.parameter_metrics.filter(r=>r.target===LAB.target);
  const names=['Ridge_5','Mean_Baseline'];
  const colors=['var(--model-ridge)','var(--model-neutral)'];
  const key=LAB.metric;
  const vals=names.map(n=>metrics.find(r=>r.model===n)?.[key]);
  addText($('#holdout-unit'),`${LAB.metric.toUpperCase()} ${labUnit()}`);
  if(vals.some(v=>!Number.isFinite(v))){root.append(toNode('p','table-note','Held-out model and baseline comparison not available.'));return;}
  const max=Math.max(...vals);
  names.forEach((name,i)=>{
    const row=toNode('div','outcome-item');row.append(toNode('div','outcome-label',i===0?'Ridge · 5':'Mean predictor'));
    const track=toNode('div','outcome-track');const fill=toNode('div','outcome-fill');fill.style.background=colors[i];fill.style.width=`${100*vals[i]/max}%`;track.append(fill);
    row.append(track,toNode('strong','',labNumber(vals[i])));root.append(row);
  });
  const better=vals[0]<vals[1];
  const tail=toNode('p','lab-micro',better?`Ridge's ${LAB.metric.toUpperCase()} is ${format((vals[1]-vals[0])/vals[1]*100,1)}% lower than the mean baseline.`:`Ridge's ${LAB.metric.toUpperCase()} is ${format((vals[0]-vals[1])/vals[1]*100,1)}% higher than the mean baseline.`);
  root.append(tail);
}
function labDrawTakeaway(){
  const best=labRows().slice().filter(r=>Number.isFinite(labScore(r))).sort((a,b)=>labScore(a)-labScore(b))[0];
  if(!best)return;
  const label=labModel(best.model)?.label||best.model;
  const d=DATA.development.development_vle||[];
  const winner=d.filter(r=>Number.isFinite(r.macro_mean_molecule_ape_percent)).slice().sort((a,b)=>a.macro_mean_molecule_ape_percent-b.macro_mean_molecule_ape_percent)[0];
  let takeaway=`For ${TARGET_LABELS[LAB.target].label}, the lowest observed ${LAB.metric.toUpperCase()} belongs to ${label} (${LAB_FEATURES.find(f=>f.key===best.features).n} features).`;
  if(winner)takeaway+=` But on the separate development thermodynamic comparison, ${DEV_MODELS.find(x=>x.key===winner.model)?.label} had the lowest mean saturation-pressure error (${pc(winner.macro_mean_molecule_ape_percent)}). The best parameter predictor need not be the best EOS predictor.`;
  addText($('#lab-takeaway-text'),takeaway);
}
function labRender(){
  labDrawChart();labDrawCollisions();labDrawHeatmap();labDrawOverfit();labDrawProperty();labDrawHoldout();labDrawTakeaway();
}
function labDownloadCSV(){
  const rows=labRows();const fields=['features','n_features','model','target','cv_mae_mean','cv_mae_std','cv_rmse_mean','cv_rmse_std'];
  const content=[fields.join(','),...rows.map(row=>fields.map(k=>`"${String(row[k]??'').replaceAll('"','""')}"`).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));
  const a=toNode('a');a.href=url;a.download=`es335_cv_${LAB.target}.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function initLab(){
  const e=DATA.experiments;
  if(!e||!Array.isArray(e.cv_models)||!e.cv_models.length||!Array.isArray(e.collisions)||e.collisions.length!==4){
    const warning=$('#model-lab-error');warning.hidden=false;addText(warning,'Model-comparison charts cannot load because the exported results.json does not include the experiment records. From the project root run: python src/export_website_data.py using your virtual environment, then refresh with Ctrl+F5. Existing final-test results are not affected.');
    let loading=$('#model-lab-loading');
    if(!loading){loading=toNode('div','model-lab-loading');loading.id='model-lab-loading';$('#model-lab-content')?.prepend(loading);}
    addText(loading,'The experiment charts are unavailable until the data exporter is updated. This is not a zero-error result.');
    return;
  }
  $('#lab-target').addEventListener('change',ev=>{LAB.target=ev.target.value;labRender();});
  $('#lab-metric').addEventListener('change',ev=>{LAB.metric=ev.target.value;labRender();});
  $('#lab-property').addEventListener('change',ev=>{LAB.property=ev.target.value;labRender();});
  $('#download-cv').addEventListener('click',labDownloadCSV);
  const loading=$('#model-lab-loading');if(loading)loading.remove();
  labRender();
}


// Appearance preference is not allowed to block any scientific content.
function initTheme(){
  const root=document.documentElement;
  const button=$('#theme-toggle');
  if(!button)return;
  const apply=(dark,save)=>{
    root.dataset.theme=dark?'dark':'light';
    button.setAttribute('aria-pressed',String(dark));
    button.setAttribute('aria-label',dark?'Switch to light theme':'Switch to dark theme');
    const label=$('#theme-label'),icon=$('#theme-icon');
    if(label)label.textContent=dark?'Light mode':'Dark mode';
    if(icon)icon.textContent=dark?'☀':'☾';
    const meta=$('#site-theme-color');if(meta)meta.content=dark?'#0b111b':'#f7f9f7';
    if(save)try{localStorage.setItem('es335-theme',dark?'dark':'light');}catch(_){}
  };
  apply(root.dataset.theme==='dark',false);
  button.addEventListener('click',()=>apply(root.dataset.theme!=='dark',true));
}

// Histogram is computed from structure descriptors only. No fitted labels or
// held-out targets enter this dataset overview.
let PROFILE_MODE='carbon_count';
function renderDatasetProfile(){
  const root=$('#dataset-profile-chart');
  if(!root)return;
  root.replaceChildren();
  const data=DATA?.dataset_profile;
  if(!data||data.total!==194||!Array.isArray(data[PROFILE_MODE])||!data[PROFILE_MODE].length){
    root.append(toNode('p','table-note','Dataset histogram not available in this export. Re-run the upgraded data exporter, then refresh. The study still contains 194 curated molecules.'));
    return;
  }
  const rows=data[PROFILE_MODE].slice();
  const max=Math.max(...rows.map(r=>r.count),1);
  const box=toNode('div','profile-bars');
  rows.forEach(item=>{
    const col=toNode('div','profile-bar-cell');
    const value=toNode('span','profile-bar-count',String(item.count));
    const track=toNode('div','profile-bar-track');
    const fill=toNode('div','profile-bar-fill');fill.style.height=`${Math.max(5,100*item.count/max)}%`;
    const name=toNode('span','profile-bar-label',String(item.value));
    track.append(fill);
    col.append(value,track,name);
    col.title=`${item.count} molecules with ${item.value} ${PROFILE_MODE==='carbon_count'?'carbon atoms':'rings'}`;
    box.append(col);
  });
  root.append(box);
  const carbon=PROFILE_MODE==='carbon_count';
  root.setAttribute('aria-label',carbon?'Molecule counts by carbon number, three through twenty':'Molecule counts by number of rings');
  addText($('#dataset-profile-note'),carbon?'194 curated saturated hydrocarbons, grouped by number of carbon atoms. Source: processed molecular descriptors.':'194 curated saturated hydrocarbons, grouped by ring count. Zero indicates acyclic compounds.');
  const carbonBtn=$('#profile-carbon'),ringBtn=$('#profile-rings');
  carbonBtn?.classList.toggle('selected',carbon); ringBtn?.classList.toggle('selected',!carbon);
  carbonBtn?.setAttribute('aria-pressed',String(carbon));ringBtn?.setAttribute('aria-pressed',String(!carbon));
}
function initProfile(){
  $('#profile-carbon')?.addEventListener('click',()=>{PROFILE_MODE='carbon_count';renderDatasetProfile();});
  $('#profile-rings')?.addEventListener('click',()=>{PROFILE_MODE='ring_count';renderDatasetProfile();});
  renderDatasetProfile();
}
function initOutlier(){
  const m=DATA.per_molecule.find(r=>r.name.toLowerCase()==='cycloheptane')||DATA.per_molecule.slice().sort((a,b)=>b.mean_pressure_ape_percent-a.mean_pressure_ape_percent)[0];
  if(!m)return;
  addText($('#outlier-name'),m.name);
  addText($('#outlier-pressure'),pc(m.mean_pressure_ape_percent));
  addText($('#outlier-density'),pc(m.mean_density_ape_percent));
  addText($('#outlier-explanation'),`A held-out hydrocarbon with ${pc(m.mean_pressure_ape_percent)} mean pressure disagreement, versus ${pc(m.mean_density_ape_percent)} for liquid density. Convergence does not equal accuracy; the plot exposes all four evaluated states.`);
  $('#inspect-outlier')?.addEventListener('click',()=>selectMolecule(m.smiles,true));
}
function showLabIssue(err){
  console.error('Model-lab display problem',err);
  const warning=$('#model-lab-error');warning.hidden=false;
  addText(warning,`The experiment graphs could not be rendered: ${err.message}. Check results.json and refresh. The source CSVs and final-test results have not been changed.`);
  const loading=$('#model-lab-loading');if(loading)addText(loading,'Chart error. See the message above for details.');
}

async function init(){
  initTheme();
  try{
    const response=await fetch('data/results.json',{cache:'no-store'});
    if(!response.ok)throw new Error(`data/results.json returned HTTP ${response.status}`);
    DATA=await response.json();
    if(DATA.per_molecule.length!==39||DATA.curves.length!==156)throw new Error('Unexpected number of final-test records; export may be incomplete.');
    DATA.per_molecule.forEach(m=>MOL_BY_ID.set(m.smiles,m));
    DATA.curves.forEach(row=>{if(!POINTS_BY_MOLECULE.has(row.smiles))POINTS_BY_MOLECULE.set(row.smiles,[]);POINTS_BY_MOLECULE.get(row.smiles).push(row);});
    stats();
    const list=DATA.per_molecule.slice().sort((a,b)=>a.name.localeCompare(b.name));
    const dropdown=$('#molecule-select');
    list.forEach(m=>{const option=toNode('option','',m.name);option.value=m.smiles;dropdown.append(option);});
    dropdown.addEventListener('change',()=>selectMolecule(dropdown.value));
    $('#table-search').addEventListener('input',renderMoleculeTable);
    $('#sort-choice').addEventListener('change',renderMoleculeTable);
    $('#download-csv').addEventListener('click',downloadCSV);
    const defaultMol=list.find(m=>m.name.toLowerCase()==='cycloheptane')||list[0];
    selectMolecule(defaultMol.smiles);
    $('#app-content').hidden=false;
    // A bad optional chart must never hide every other section on the site.
    try{initLab();}catch(err){showLabIssue(err);}
    try{initProfile();}catch(err){console.error('Dataset profile failed',err);addText($('#dataset-profile-chart'),'Dataset histogram unavailable: '+err.message);}
    try{initOutlier();}catch(err){console.error('Outlier card failed',err);}
    try{development();}catch(err){console.error('Development summary failed',err);}
    try{sensitivity();}catch(err){console.error('Sensitivity table failed',err);}
  }catch(err){const alert=$('#load-error');alert.hidden=false;addText(alert,`Unable to load frozen held-out results: ${err.message}. Run python src/export_website_data.py with your project virtual environment from the repository root, then use a local HTTP server (not file://).`);showLabIssue(err);console.error(err);}
}
window.addEventListener('DOMContentLoaded',init);
