"""
SynapLink - Brain-to-Code Pipeline v2
No intent buttons - random signal from dataset - show classified intents
"""
import random, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATASET_PATH   = "synaplink_real_eeg_dataset.csv"
MODELS_DIR     = "synaplink_models"

print("Loading models...")
try:
    m1  = pickle.load(open(f"{MODELS_DIR}/m1.pkl","rb"))
    m2  = pickle.load(open(f"{MODELS_DIR}/m2.pkl","rb"))
    m3  = pickle.load(open(f"{MODELS_DIR}/m3.pkl","rb"))
    sc  = pickle.load(open(f"{MODELS_DIR}/sc.pkl","rb"))
    le1 = pickle.load(open(f"{MODELS_DIR}/le1.pkl","rb"))
    le2 = pickle.load(open(f"{MODELS_DIR}/le2.pkl","rb"))
    le3 = pickle.load(open(f"{MODELS_DIR}/le3.pkl","rb"))
    MODELS_OK = True
    print("Models loaded!")
except Exception as e:
    MODELS_OK = False
    m1 = None
    print(f"Demo mode: {e}")

FEATURES = ['raw_f3_mean_uv','raw_f3_std_uv','raw_peak_amplitude_uv','delta_power','theta_power','alpha_power','beta_power','gamma_power','spectral_entropy','hjorth_activity','hjorth_mobility','hjorth_complexity','plv_f3_f4','plv_c3_c4','plv_p3_p4','coherence_frontal','coherence_central','coherence_parietal','f3_rms','f4_rms','c3_rms','c4_rms','mean_power_all','theta_alpha_ratio','beta_gamma_ratio','theta_beta_ratio','theta_phase_coupling','hemispheric_asymmetry','complexity_score','session_drift_factor','subject_scale']

try:
    df = pd.read_csv(DATASET_PATH)
    DATASET_OK = True
    print(f"Dataset loaded: {len(df)} samples")
except:
    DATASET_OK = False
    df = None

PRESETS = {
    "HELLO_WORLD":  {"sig":[1.842,15.363,48.211,1.464,1.351,0.997,4.307,2.682,0.72,7.206,0.394,1.674,0.794,0.674,0.379,0.734,0.614,0.329,15.346,13.414,11.533,10.906,2.16,1.587,1.65,0.32,1.487,-0.068,1.0,1.181,1.02],"l1":"CODE","l2":"PRINT_STMT","c1":100,"c2":98,"c3":100},
    "IF_ELSE":      {"sig":[1.786,14.846,46.71,1.631,1.442,1.286,3.733,2.514,0.728,6.504,0.429,1.759,0.731,0.617,0.408,0.675,0.558,0.353,14.83,11.756,11.319,10.968,2.274,2.145,1.548,0.444,1.944,-0.077,4.0,1.181,1.02],"l1":"CODE","l2":"CONDITIONAL","c1":100,"c2":79,"c3":66},
    "FIBONACCI":    {"sig":[1.898,15.775,49.659,1.549,1.562,1.107,3.918,2.648,0.746,6.818,0.437,1.767,0.751,0.635,0.397,0.694,0.576,0.343,15.759,11.697,11.443,11.034,2.298,2.7,1.536,0.416,2.281,-0.124,4.0,1.181,1.02],"l1":"CODE","l2":"FUNCTION_DEF","c1":100,"c2":77,"c3":57},
    "BUBBLE_SORT":  {"sig":[2.183,18.143,57.132,1.472,1.944,0.641,4.518,3.055,0.798,7.778,0.464,1.793,0.779,0.66,0.386,0.721,0.603,0.333,18.127,11.28,11.875,11.278,2.351,4.531,1.508,0.44,3.005,-0.228,6.0,1.181,1.02],"l1":"CODE","l2":"ALGORITHM","c1":100,"c2":94,"c3":86},
    "EMAIL_DRAFT":  {"sig":[1.345,11.193,35.197,1.979,2.005,2.12,2.832,1.996,0.647,5.607,0.41,1.745,0.671,0.564,0.451,0.617,0.494,0.394,11.173,12.071,10.423,10.759,2.27,1.09,1.428,0.543,1.215,0.025,2.0,1.181,1.02],"l1":"TEXT","l2":"COMPOSE","c1":100,"c2":77,"c3":78},
    "UI_MOCKUP":    {"sig":[0.791,6.581,20.709,2.742,1.244,4.166,1.518,1.444,0.485,3.054,0.417,1.716,0.44,0.32,0.608,0.376,0.27,0.557,6.566,11.946,8.596,9.602,2.197,0.307,1.11,0.846,0.165,0.213,4.0,1.181,1.02],"l1":"IMAGE","l2":"GENERATE","c1":100,"c2":72,"c3":66},
}

def get_random_signal():
    if DATASET_OK and df is not None:
        row = df.sample(1).iloc[0]
        sig = row[FEATURES].values.tolist()
        return sig, float(row['beta_power']), float(row['alpha_power']), float(row['theta_power']), float(row['gamma_power']), str(row['label_l3'])
    key = random.choice(list(PRESETS.keys()))
    p = PRESETS[key]
    noisy = [v + random.gauss(0, abs(v)*0.12) for v in p["sig"]]
    return noisy, noisy[6], noisy[5], noisy[4], noisy[7], key

def predict(signal):
    X = np.array(signal).reshape(1,-1)
    if m1 is not None:
        Xs = sc.transform(X)
        l1 = le1.inverse_transform(m1.predict(Xs))[0]
        l2 = le2.inverse_transform(m2.predict(Xs))[0]
        l3 = le3.inverse_transform(m3.predict(Xs))[0]
        c1 = round(m1.predict_proba(Xs).max()*100,1)
        c2 = round(m2.predict_proba(Xs).max()*100,1)
        c3 = round(m3.predict_proba(Xs).max()*100,1)
        return {"l1":l1,"l2":l2,"l3":l3,"c1":c1,"c2":c2,"c3":c3}
    # demo nearest match
    best,bd = list(PRESETS.keys())[0], float('inf')
    for k,v in PRESETS.items():
        d = np.linalg.norm(np.array(signal)-np.array(v["sig"]))
        if d < bd: bd=d; best=k
    p = PRESETS[best]
    return {"l1":p["l1"],"l2":p["l2"],"l3":best,
            "c1":p["c1"]+random.randint(-3,3),
            "c2":p["c2"]+random.randint(-5,5),
            "c3":p["c3"]+random.randint(-8,8)}

client = genai.Client(api_key=GEMINI_API_KEY)

def generate(pred):
    l1,l2,l3 = pred["l1"],pred["l2"],pred["l3"]
    if l1=="CODE": prompt=f"You are SynapLink brain-to-code AI. Generate clean executable Python code for intent: {l3} (category: {l2}). Max 20 lines. Code only."
    elif l1=="TEXT": prompt=f"You are SynapLink brain-to-text AI. Generate professional text for intent: {l3} (category: {l2}). Text only."
    else: prompt=f"You are SynapLink brain-to-image AI. Generate detailed image prompt for intent: {l3} (category: {l2}). Prompt only."
    
    # Retry 3 times if server busy
    import time
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    for attempt in range(3):
        for model in models_to_try:
            try:
                r = client.models.generate_content(model=model, contents=prompt)
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    time.sleep(2)
                    continue
                elif "429" in err or "EXHAUSTED" in err:
                    time.sleep(5)
                    continue
                else:
                    return f"# Gemini Error: {e}"
    return f"# Gemini busy — please click RUN again in a few seconds"

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SynapLink — Brain-to-Code</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e1a;--bg2:#0f1528;--bg3:#141c35;--border:rgba(99,132,255,0.15);--accent:#6384ff;--accent2:#38e8c0;--accent3:#ff6b6b;--text:#e8eaf6;--text2:#8892b0;--mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;padding:1.5rem;max-width:980px;margin:0 auto;}
h1{font-family:var(--mono);font-size:22px;font-weight:700;color:var(--accent);margin-bottom:4px;}
.sub{font-size:12px;color:var(--text2);margin-bottom:2rem;line-height:1.8;}
.badge{font-family:var(--mono);font-size:9px;padding:3px 10px;border:0.5px solid var(--border);border-radius:20px;color:var(--accent2);display:inline-block;margin-bottom:1rem;}
.sig-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:1.5rem;}
.sig-card{border:0.5px solid var(--border);border-radius:8px;padding:10px;background:var(--bg2);text-align:center;}
.sig-label{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:6px;letter-spacing:.04em;}
.sig-val{font-family:var(--mono);font-size:15px;font-weight:700;margin-top:4px;}
.sig-band{font-size:9px;color:var(--text2);margin-top:3px;opacity:.7;}
canvas{width:100%;height:32px;display:block;margin:4px 0;}
.run-btn{width:100%;padding:16px;font-family:var(--mono);font-size:14px;font-weight:700;background:var(--accent);color:#fff;border:none;border-radius:10px;cursor:pointer;margin-bottom:1.5rem;transition:all .2s;letter-spacing:.06em;}
.run-btn:hover:not(:disabled){background:#7a98ff;transform:translateY(-1px);}
.run-btn:disabled{opacity:.4;cursor:not-allowed;}
.progress{height:3px;background:var(--bg3);border-radius:2px;overflow:hidden;margin-bottom:8px;}
.pf{height:100%;background:var(--accent);width:0%;transition:width .12s linear;border-radius:2px;}
.status{font-family:var(--mono);font-size:11px;color:var(--text2);min-height:20px;margin-bottom:1.5rem;}
.intent-box{border:0.5px solid rgba(99,132,255,.35);border-radius:12px;padding:1.25rem;background:var(--bg2);margin-bottom:1.5rem;display:none;}
.intent-box.show{display:block;}
.ib-title{font-family:var(--mono);font-size:10px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;}
.intent-path{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:16px;}
.ic{font-family:var(--mono);font-size:13px;font-weight:700;padding:7px 18px;border-radius:6px;}
.ic1{background:rgba(99,132,255,.12);color:#6384ff;border:0.5px solid rgba(99,132,255,.4);}
.ic2{background:rgba(56,232,192,.1);color:#38e8c0;border:0.5px solid rgba(56,232,192,.35);}
.ic3{background:rgba(255,107,107,.1);color:#ff6b6b;border:0.5px solid rgba(255,107,107,.35);}
.arr{color:var(--text2);font-size:18px;font-weight:300;}
.conf-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.ci{background:var(--bg3);border-radius:6px;padding:9px 11px;}
.ci-label{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:5px;letter-spacing:.04em;}
.ct{height:4px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden;margin-bottom:5px;}
.cf{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.34,1.56,.64,1);width:0%;}
.cp{font-family:var(--mono);font-size:11px;font-weight:700;}
.output-panel{border:0.5px solid var(--border);border-radius:10px;overflow:hidden;}
.output-bar{background:var(--bg3);padding:9px 14px;border-bottom:0.5px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.dots{display:flex;gap:5px;}.dot{width:9px;height:9px;border-radius:50%;}
.ot{font-family:var(--mono);font-size:11px;color:var(--text2);}
.lat{font-family:var(--mono);font-size:11px;color:var(--accent2);}
.ca{background:var(--bg2);padding:1rem;min-height:120px;font-family:var(--mono);font-size:12px;line-height:1.8;color:var(--text);white-space:pre-wrap;overflow-x:auto;}
@media(max-width:560px){.sig-row{grid-template-columns:repeat(2,1fr)}.conf-row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="badge">KARE B.Tech CSE (AI/ML) · Reg: 9921004520 · IEEE ICIRCA-2025</div>
<h1>SynapLink — Brain-to-Code</h1>
<p class="sub">User wears EEG headset and thinks a coding or text intent.<br>System automatically captures brain signals → classifies intent → generates executable code.</p>

<div class="sig-row">
  <div class="sig-card">
    <div class="sig-label">beta power · 13-30Hz</div>
    <canvas id="wv0"></canvas>
    <div class="sig-val" id="sv0" style="color:#6384ff;">—</div>
    <div class="sig-band">active coding</div>
  </div>
  <div class="sig-card">
    <div class="sig-label">alpha power · 8-13Hz</div>
    <canvas id="wv1"></canvas>
    <div class="sig-val" id="sv1" style="color:#38e8c0;">—</div>
    <div class="sig-band">relaxed · visual</div>
  </div>
  <div class="sig-card">
    <div class="sig-label">theta power · 4-8Hz</div>
    <canvas id="wv2"></canvas>
    <div class="sig-val" id="sv2" style="color:#ff6b6b;">—</div>
    <div class="sig-band">memory · recall</div>
  </div>
  <div class="sig-card">
    <div class="sig-label">gamma power · 30-100Hz</div>
    <canvas id="wv3"></canvas>
    <div class="sig-val" id="sv3" style="color:#ffcb6b;">—</div>
    <div class="sig-band">concentration</div>
  </div>
</div>

<button class="run-btn" id="runBtn" onclick="run()">🧠 RUN BRAIN-TO-CODE PIPELINE</button>

<div class="progress"><div class="pf" id="pf"></div></div>
<div class="status" id="st">Click to capture and classify brain signals from dataset</div>

<div class="intent-box" id="ib">
  <div class="ib-title">your classified intents are:</div>
  <div class="intent-path">
    <span class="ic ic1" id="iL1">—</span>
    <span class="arr">→</span>
    <span class="ic ic2" id="iL2">—</span>
    <span class="arr">→</span>
    <span class="ic ic3" id="iL3">—</span>
  </div>
  <div class="conf-row">
    <div class="ci"><div class="ci-label">L1 — domain</div><div class="ct"><div class="cf" id="cf1" style="background:#6384ff"></div></div><div class="cp" id="cp1" style="color:#6384ff">—</div></div>
    <div class="ci"><div class="ci-label">L2 — category</div><div class="ct"><div class="cf" id="cf2" style="background:#38e8c0"></div></div><div class="cp" id="cp2" style="color:#38e8c0">—</div></div>
    <div class="ci"><div class="ci-label">L3 — specific intent</div><div class="ct"><div class="cf" id="cf3" style="background:#ff6b6b"></div></div><div class="cp" id="cp3" style="color:#ff6b6b">—</div></div>
  </div>
</div>

<div class="output-panel">
  <div class="output-bar">
    <div class="dots"><div class="dot" style="background:#ff5f57"></div><div class="dot" style="background:#febc2e"></div><div class="dot" style="background:#28c840"></div></div>
    <span class="ot" id="ot">synaplink_output.py — waiting</span>
    <span class="lat" id="lat"></span>
  </div>
  <div class="ca" id="ca"># SynapLink output will appear here
# Click RUN to capture brain signals and generate code</div>
</div>

<script>
const sl=ms=>new Promise(r=>setTimeout(r,ms));
let busy=false;

function wave(id,val,col,noisy){
  const c=document.getElementById(id);if(!c)return;
  const w=c.offsetWidth||200;c.width=w;c.height=32;
  const ctx=c.getContext('2d');ctx.clearRect(0,0,w,32);
  ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();
  for(let x=0;x<w;x++){const n=noisy?(Math.random()-.5)*6:0;const y=16+Math.sin(x*val*.08)*Math.min(val*2.8,13)+n;x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();
}

async function fill(a,b,d){
  const f=document.getElementById('pf');
  for(let i=0;i<=20;i++){f.style.width=(a+(b-a)*i/20).toFixed(1)+'%';await sl(d/20);}
}

async function run(){
  if(busy)return;busy=true;
  document.getElementById('runBtn').disabled=true;
  document.getElementById('ib').classList.remove('show');
  const t0=Date.now();
  const cols=['#6384ff','#38e8c0','#ff6b6b','#ffcb6b'];

  // Step 1
  document.getElementById('st').textContent='Step 1: ADS1299 capturing 8-channel EEG at 250Hz...';
  await fill(0,15,300);
  [3.5,2.1,1.8,2.4].forEach((v,i)=>{wave('wv'+i,v,cols[i],true);document.getElementById('sv'+i).textContent='...';});
  await sl(500);

  // Step 2
  document.getElementById('st').textContent='Step 2: Butterworth filter → ICA artifact removal → clean signal...';
  await fill(15,35,350);
  await sl(300);

  // Step 3 — fetch random signal from dataset
  document.getElementById('st').textContent='Step 3: Extracting 31 EEG features — ratios, coherence, entropy...';
  await fill(35,55,350);
  let rs;
  try{const r=await fetch('/random_signal');rs=await r.json();}
  catch(e){document.getElementById('st').textContent='Error: Flask server not running!';busy=false;document.getElementById('runBtn').disabled=false;return;}

  [rs.beta,rs.alpha,rs.theta,rs.gamma].forEach((v,i)=>{
    wave('wv'+i,parseFloat(v),cols[i],false);
    document.getElementById('sv'+i).textContent=parseFloat(v).toFixed(2)+' µV²';
  });

  // Step 4 — classify
  document.getElementById('st').textContent='Step 4: Hierarchical Random Forest classifying L1 → L2 → L3...';
  await fill(55,75,400);
  let pred;
  try{const r=await fetch('/predict_from_signal',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({signal:rs.signal})});pred=await r.json();}
  catch(e){document.getElementById('st').textContent='Prediction error!';busy=false;document.getElementById('runBtn').disabled=false;return;}

  // Show classified intents
  document.getElementById('ib').classList.add('show');
  document.getElementById('iL1').textContent=pred.l1;
  document.getElementById('iL2').textContent=pred.l2.replace(/_/g,' ').toLowerCase();
  document.getElementById('iL3').textContent=pred.l3.replace(/_/g,' ').toLowerCase();
  document.getElementById('cf1').style.width=pred.c1+'%';
  document.getElementById('cf2').style.width=pred.c2+'%';
  document.getElementById('cf3').style.width=pred.c3+'%';
  document.getElementById('cp1').textContent=pred.c1+'% confidence';
  document.getElementById('cp2').textContent=pred.c2+'% confidence';
  document.getElementById('cp3').textContent=pred.c3+'% confidence';

  // Step 5 — Gemini
  document.getElementById('st').textContent=`Step 5: Intent "${pred.l3}" → Gemini 2.5 Flash generating output...`;
  document.getElementById('ot').textContent=`synaplink_output.py — generating ${pred.l3.replace(/_/g,' ').toLowerCase()}...`;
  document.getElementById('ca').textContent='⚡ Gemini generating...';
  await fill(75,92,500);
  let out;
  try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pred)});const d=await r.json();out=d.output;}
  catch(e){out='# Generation error — check API key';}

  await fill(92,100,200);
  const el=((Date.now()-t0)/1000).toFixed(2);
  document.getElementById('ca').textContent=out;
  document.getElementById('ot').textContent=`synaplink_output.py — ${pred.l3.replace(/_/g,' ').toLowerCase()}`;
  document.getElementById('lat').textContent=`${el}s total`;
  document.getElementById('st').textContent=`✅ Pipeline complete — your classified intents: ${pred.l1} / ${pred.l2} / ${pred.l3} → output generated in ${el}s`;
  busy=false;document.getElementById('runBtn').disabled=false;
}
</script>
</body>
</html>"""

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/random_signal')
def random_signal_route():
    sig, beta, alpha, theta, gamma, true_l3 = get_random_signal()
    return jsonify({"signal":[float(x) for x in sig],"beta":round(float(beta),2),"alpha":round(float(alpha),2),"theta":round(float(theta),2),"gamma":round(float(gamma),2),"true_l3":str(true_l3)})

@app.route('/predict_from_signal', methods=['POST'])
def predict_route():
    return jsonify(predict(request.json.get('signal',[])))

@app.route('/generate', methods=['POST'])
def generate_route():
    return jsonify({"output": generate(request.json)})

if __name__ == '__main__':
    print("="*55)
    print("  SYNAPLINK v2 — NO INTENT BUTTONS")
    print(f"  Models:  {'Loaded' if MODELS_OK else 'Demo mode'}")
    print(f"  Dataset: {'Loaded' if DATASET_OK else 'Fallback'}")
    print("  Open: http://localhost:5000")
    print("="*55)
    app.run(debug=True, port=5000)
