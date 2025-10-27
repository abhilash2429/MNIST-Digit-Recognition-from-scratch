const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predVal = document.getElementById('predVal');
const confidencesDiv = document.getElementById('confidences');

let drawing = false;
let last = {x:0,y:0};

ctx.fillStyle = 'white';
ctx.fillRect(0,0,canvas.width, canvas.height);
ctx.lineWidth = 24;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

function getPos(e){
  const rect = canvas.getBoundingClientRect();
  return {x: (e.clientX - rect.left), y: (e.clientY - rect.top)}
}

canvas.addEventListener('pointerdown', (e)=>{drawing=true; last = getPos(e)});
canvas.addEventListener('pointerup', ()=>{drawing=false});
canvas.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
});

clearBtn.addEventListener('click', ()=>{ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height); predVal.textContent='-'; confidencesDiv.innerHTML='';});

predictBtn.addEventListener('click', async ()=>{
  const dataUrl = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:dataUrl})});
  const j = await res.json();
  if(j.error){alert(j.error); return}
  predVal.textContent = j.prediction;
  confidencesDiv.innerHTML = '';
  for (const [i, c] of j.confidences.entries()) {
    const el = document.createElement('div'); el.className = 'conf';
    const label = document.createElement('div'); label.className = 'label'; label.textContent = i;
    const barWrap = document.createElement('div'); barWrap.className = 'bar';
    const bar = document.createElement('i'); bar.style.width = Math.round(c * 100) + '%';
    barWrap.appendChild(bar);
    const num = document.createElement('div'); num.textContent = (c * 100).toFixed(1) + '%';
    el.appendChild(label);
    el.appendChild(barWrap);
    el.appendChild(num);
    confidencesDiv.appendChild(el);
  }
});
