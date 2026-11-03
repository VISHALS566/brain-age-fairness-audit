"""
Brain-Age Fairness Audit — Research Dashboard
Flask app: results dashboard + live inference + dementia clinical panel.
"""
import io, json, os, re, sys, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model_v2 import BrainAgeResNet, K_BINS, BIN_CENTERS, decode, build_model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUT_DIR     = BASE_DIR / "outputs" / "unified"
MODEL_PATH  = OUT_DIR / "best_model_unified.pt"
PRED_CSV    = OUT_DIR / "unified_predictions.csv"
FAIR_CSV    = OUT_DIR / "fairness_report.csv"
HIST_CSV    = OUT_DIR / "training_history.csv"
META_JSON   = OUT_DIR / "meta_stats.json"
OASIS_CSV   = Path(r"E:\code\outputs\brain_age_v2\manifest.csv")
OASIS_ROOT  = Path(r"E:\code\OASIS-1")

TARGET_SHAPE = (80, 80, 96)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ── Global model (loaded once) ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = None
META_STATS = None

def get_model():
    global MODEL, META_STATS
    if MODEL is None:
        MODEL = BrainAgeResNet(in_ch=1, n_meta=3, n_bins=K_BINS).to(DEVICE)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        MODEL.eval()
        if META_JSON.exists():
            META_STATS = json.load(open(META_JSON))
    return MODEL

# ── Volume pre-processing ──────────────────────────────────────────────────────
def _to_tensor(vol):
    vol = vol.astype("float32")
    mask = vol > 0
    if mask.sum() < 500:
        return torch.zeros((1,) + TARGET_SHAPE)
    lo, hi = np.percentile(vol[mask], [1, 99])
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    vol[~mask] = 0.0
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=TARGET_SHAPE, mode="area")
    return torch.nan_to_num(t.squeeze(0), nan=0.0).float()

def preprocess_pkl(data):
    obj = pickle.loads(data)
    vol = np.array(obj[0], dtype="float32")
    vol[~np.isfinite(vol)] = 0.0
    return _to_tensor(vol)

def preprocess_nii(data):
    fh  = nib.FileHolder(fileobj=io.BytesIO(data))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    vol = np.squeeze(img.get_fdata()).astype("float32")
    vol = np.ascontiguousarray(np.flip(np.flip(np.transpose(vol, (0, 2, 1)), 0), 1))
    return _to_tensor(vol)

def preprocess_hdr_path(hdr_path):
    """Load OASIS .hdr/.img from filesystem path."""
    vol = np.squeeze(nib.load(hdr_path).get_fdata()).astype("float32")
    vol = np.ascontiguousarray(np.flip(np.flip(np.transpose(vol, (0, 2, 1)), 0), 1))
    return _to_tensor(vol)

def build_meta_tensor(sex, dataset="IXI", site="WashU"):
    sex_v  = {"male": 1.0, "female": 0.0}.get(sex.lower(), 0.5)
    ds_v   = 1.0 if dataset.upper() == "IXI" else 0.0
    site_map = {"washu": 0.0, "hammersmith": 0.25, "guys": 0.5, "kcl": 0.75}
    site_v = site_map.get(site.lower(), 0.0)
    x = np.array([[sex_v, ds_v, site_v]], dtype="float32")
    if META_STATS:
        x = (x - np.array(META_STATS["mean"])) / np.array(META_STATS["std"])
    return torch.from_numpy(x).float()

def _run_inference(vol_tensor, sex, dataset, site="WashU"):
    model  = get_model()
    vol_t  = vol_tensor.unsqueeze(0).to(DEVICE)
    meta_t = build_meta_tensor(sex, dataset, site).to(DEVICE)
    with torch.no_grad():
        logits   = model(vol_t, meta_t)
        logits_f = model(vol_t.flip(4), meta_t)
        pred_age = float(0.5 * (decode(logits) + decode(logits_f)).item())
    return pred_age

# ── Chart builders ─────────────────────────────────────────────────────────────
_BG = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
       "font": {"color": "#e2e8f0"}}
_GRID = "#1e293b"

def chart_training():
    df = pd.read_csv(HIST_CSV)
    return {
        "data": [
            {"x": df.epoch.tolist(), "y": df.train_loss.tolist(),
             "name": "Train Loss", "type": "scatter", "mode": "lines",
             "line": {"color": "#4f8ef7", "width": 2}},
            {"x": df.epoch.tolist(), "y": df.val_mae.tolist(),
             "name": "Val MAE (years)", "type": "scatter", "mode": "lines",
             "line": {"color": "#00d4ff", "width": 2}, "yaxis": "y2"},
        ],
        "layout": {**_BG,
            "xaxis": {"title": "Epoch", "gridcolor": _GRID},
            "yaxis": {"title": "Train Loss", "gridcolor": _GRID},
            "yaxis2": {"title": "Val MAE (years)", "overlaying": "y", "side": "right", "gridcolor": _GRID},
            "legend": {"x": 0.7, "y": 0.95},
            "margin": {"t": 20, "b": 40, "l": 60, "r": 60},
        }
    }

def chart_scatter():
    df = pd.read_csv(PRED_CSV)
    colors = {"IXI": "#4f8ef7", "OASIS": "#f59e0b"}
    data = []
    for ds, grp in df.groupby("dataset"):
        data.append({
            "x": grp.age.tolist(), "y": grp.prediction.tolist(),
            "name": ds, "type": "scatter", "mode": "markers",
            "marker": {"color": colors.get(ds, "#8b5cf6"), "size": 6, "opacity": 0.75},
            "text": [f"Subject: {s}<br>True: {a:.1f}y<br>Pred: {p:.1f}y<br>BAG: {b:+.1f}y"
                     for s, a, p, b in zip(grp.subject_id, grp.age, grp.prediction, grp.bag)],
            "hovertemplate": "%{text}<extra></extra>",
        })
    mn, mx = float(df.age.min())-2, float(df.age.max())+2
    data.append({"x": [mn, mx], "y": [mn, mx], "name": "Perfect",
                 "type": "scatter", "mode": "lines",
                 "line": {"color": "#64748b", "dash": "dash", "width": 1}})
    return {"data": data, "layout": {**_BG,
        "xaxis": {"title": "Chronological Age (years)", "gridcolor": _GRID},
        "yaxis": {"title": "Predicted Brain Age (years)", "gridcolor": _GRID},
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"t": 20, "b": 50, "l": 60, "r": 20}}}

def chart_bag_by_group(col, palette=None):
    df = pd.read_csv(PRED_CSV)
    palette = palette or ["#4f8ef7","#00d4ff","#8b5cf6","#f59e0b","#10b981"]
    data = [{"y": grp.bag.tolist(), "name": str(name), "type": "violin",
              "box": {"visible": True}, "meanline": {"visible": True},
              "fillcolor": palette[i % len(palette)],
              "line": {"color": palette[i % len(palette)]}, "opacity": 0.7, "points": "outliers"}
             for i, (name, grp) in enumerate(df.groupby(col))]
    return {"data": data, "layout": {**_BG,
        "yaxis": {"title": "Brain Age Gap (years)", "gridcolor": _GRID,
                  "zeroline": True, "zerolinecolor": "#f87171", "zerolinewidth": 1.5},
        "xaxis": {"gridcolor": _GRID}, "showlegend": False,
        "margin": {"t": 20, "b": 50, "l": 60, "r": 20}}}

def chart_mae_bar():
    df = pd.read_csv(FAIR_CSV)
    colors = {"Sex": "#4f8ef7", "Site": "#8b5cf6", "Dataset": "#f59e0b"}
    data = [{"x": sub.group_value.tolist(), "y": sub.MAE.tolist(), "name": gt, "type": "bar",
              "marker": {"color": colors.get(gt, "#64748b")},
              "text": [f"{v:.2f}y" for v in sub.MAE], "textposition": "outside"}
             for gt in ["Sex","Site","Dataset"]
             for sub in [df[df.group_type == gt]]]
    return {"data": data, "layout": {**_BG,
        "yaxis": {"title": "MAE (years)", "gridcolor": _GRID},
        "xaxis": {"gridcolor": _GRID}, "barmode": "group",
        "legend": {"x": 0.7, "y": 0.98}, "margin": {"t": 20, "b": 60, "l": 60, "r": 20}}}

def chart_age_decade_mae():
    sub = pd.read_csv(FAIR_CSV).query("group_type == 'Age_decade'").sort_values("group_value")
    return {"data": [
        {"x": sub.group_value.tolist(), "y": sub.MAE.tolist(), "name": "MAE", "type": "bar",
         "marker": {"color": sub.MAE.tolist(), "colorscale": "RdYlGn_r",
                    "showscale": True, "cmin": 4, "cmax": 10},
         "text": [f"{v:.2f}y" for v in sub.MAE], "textposition": "outside"},
        {"x": sub.group_value.tolist(), "y": sub.mean_BAG.tolist(), "name": "Mean BAG",
         "type": "scatter", "mode": "lines+markers",
         "marker": {"color": "#f59e0b", "size": 8},
         "line": {"color": "#f59e0b", "width": 2}, "yaxis": "y2"},
    ], "layout": {**_BG,
        "xaxis": {"title": "Age Decade", "gridcolor": _GRID},
        "yaxis": {"title": "MAE (years)", "gridcolor": _GRID},
        "yaxis2": {"title": "Mean BAG (years)", "overlaying": "y", "side": "right",
                   "zeroline": True, "zerolinecolor": "#f87171", "gridcolor": _GRID},
        "legend": {"x": 0.02, "y": 0.98}, "margin": {"t": 20, "b": 60, "l": 60, "r": 60}}}

# ── Dementia charts ────────────────────────────────────────────────────────────
def _merged_clinical():
    pred = pd.read_csv(PRED_CSV)
    op   = pred[pred.dataset == "OASIS"].copy()
    if OASIS_CSV.exists():
        m  = pd.read_csv(OASIS_CSV).rename(columns={"SUBJECT_ID": "subject_id"})
        op = op.merge(m[["subject_id","CDR","MMSE"]], on="subject_id", how="left")
    return op

def chart_cdr_bag():
    df = _merged_clinical().dropna(subset=["CDR"])
    cdr_labels = {0.0:"Healthy\n(CDR 0)", 0.5:"Very Mild\n(CDR 0.5)",
                  1.0:"Mild\n(CDR 1)", 2.0:"Moderate\n(CDR 2)"}
    palette    = {0.0:"#10b981", 0.5:"#f59e0b", 1.0:"#ef4444", 2.0:"#7f1d1d"}
    rows = sorted([{"label": cdr_labels.get(c, f"CDR {c}"), "cdr": c,
                    "mean_bag": float(g.bag.mean()), "std_bag": float(g.bag.std()),
                    "n": len(g), "color": palette.get(c, "#8b5cf6")}
                   for c, g in df.groupby("CDR")], key=lambda r: r["cdr"])
    return {"data": [{"x": [r["label"] for r in rows], "y": [r["mean_bag"] for r in rows],
        "error_y": {"type":"data","array":[r["std_bag"] for r in rows],
                    "visible":True,"color":"#94a3b8"},
        "type":"bar", "name":"Mean BAG", "marker":{"color":[r["color"] for r in rows]},
        "text":[f"N={r['n']}<br>BAG={r['mean_bag']:+.2f}y" for r in rows],
        "textposition":"outside"}],
    "layout": {**_BG,
        "yaxis":{"title":"Mean Brain Age Gap (years)","gridcolor":_GRID,
                 "zeroline":True,"zerolinecolor":"#f87171","zerolinewidth":2},
        "xaxis":{"gridcolor":_GRID},"showlegend":False,
        "margin":{"t":30,"b":60,"l":70,"r":20}}}

def chart_mmse_bag():
    df = _merged_clinical().dropna(subset=["MMSE","bag"])
    pal = {0.0:"#10b981", 0.5:"#f59e0b", 1.0:"#ef4444", 2.0:"#7f1d1d"}
    data = [{"x": g.MMSE.tolist(), "y": g.bag.tolist(), "name": f"CDR {c}",
              "type":"scatter","mode":"markers",
              "marker":{"color":pal.get(c,"#8b5cf6"),"size":9,"opacity":0.8},
              "text":[f"Subject:{s}<br>Age:{a:.0f}y<br>Pred:{p:.1f}y<br>BAG:{b:+.1f}y<br>MMSE:{mm:.0f}"
                      for s,a,p,b,mm in zip(g.subject_id,g.age,g.prediction,g.bag,g.MMSE)],
              "hovertemplate":"%{text}<extra></extra>"}
             for c,g in df.groupby("CDR")]
    corr = float(np.corrcoef(df.MMSE, df.bag)[0,1]) if len(df)>5 else 0
    return {"data":data,"layout":{**_BG,
        "xaxis":{"title":"MMSE Score (cognitive function)","gridcolor":_GRID,"autorange":"reversed"},
        "yaxis":{"title":"Brain Age Gap — BAG (years)","gridcolor":_GRID,
                 "zeroline":True,"zerolinecolor":"#f87171"},
        "legend":{"x":0.02,"y":0.98},
        "annotations":[{"x":0.98,"y":0.04,"xref":"paper","yref":"paper",
                         "text":f"r = {corr:.3f}  (MMSE ↑ → BAG ↓)",
                         "showarrow":False,"font":{"color":"#94a3b8","size":12},"align":"right"}],
        "margin":{"t":20,"b":60,"l":70,"r":20}}}

# ── Key metrics & tables ───────────────────────────────────────────────────────
def get_key_metrics():
    pred = pd.read_csv(PRED_CSV)
    fair = pd.read_csv(FAIR_CSV)
    overall = fair[fair.group_type == "Overall"].iloc[0]
    hist    = pd.read_csv(HIST_CSV)
    return {
        "overall_mae":     round(float(overall.MAE), 2),
        "overall_r2":      round(float(r2_score(pred.age, pred.prediction)), 3),
        "within5":         round(float((pred.abs_error <= 5).mean() * 100), 1),
        "within10":        round(float((pred.abs_error <= 10).mean() * 100), 1),
        "n_subjects":      len(pred),
        "ixi_mae":         round(float(pred[pred.dataset=="IXI"]["abs_error"].mean()), 2),
        "oasis_mae":       round(float(pred[pred.dataset=="OASIS"]["abs_error"].mean()), 2),
        "best_val_mae":    round(float(hist.val_mae.min()), 2),
        "train_subjects":  683,
        "total_subjects":  937,
        "n_sites":         4,
        "baseline_mae":    15.60,
        "improvement_pct": round((1 - float(overall.MAE) / 15.60) * 100, 1),
    }

def get_fairness_table():
    return pd.read_csv(FAIR_CSV).round(3).to_dict("records")

def get_oasis_clinical():
    df = _merged_clinical()
    cdr_labels = {0.0:"Healthy", 0.5:"Very Mild", 1.0:"Mild", 2.0:"Moderate"}
    cdr_groups = [{"cdr":float(c),"label":cdr_labels.get(c,f"CDR {c}"),
                   "n":len(g),"mean_bag":round(float(g.bag.mean()),2),
                   "std_bag":round(float(g.bag.std()),2),
                   "mae":round(float(g.abs_error.mean()),2),
                   "within5":round(float((g.abs_error<=5).mean()*100),1)}
                  for c,g in df.dropna(subset=["CDR"]).groupby("CDR")]
    mm = df.dropna(subset=["MMSE","bag"])
    corr = float(np.corrcoef(mm.MMSE, mm.bag)[0,1]) if len(mm)>5 else None
    cols = [c for c in ["subject_id","age","prediction","bag","abs_error","sex","CDR","MMSE"] if c in df.columns]
    table = df[cols].sort_values(["CDR","bag"],ascending=[False,False]).round(2).to_dict("records")
    return {
        "cdr_groups": cdr_groups,
        "mmse_corr":  round(corr,3) if corr else None,
        "n_demented": int((df["CDR"]>0).sum()) if "CDR" in df.columns else 0,
        "n_total":    len(df),
        "subjects":   table,
    }

def get_oasis_subjects():
    if not OASIS_CSV.exists():
        return []
    m = pd.read_csv(OASIS_CSV).rename(columns={
        "SUBJECT_ID":"subject_id","AGE":"age","SEX":"sex",
        "CDR":"cdr","MMSE":"mmse","FILE_PATH":"file_path","SPLIT":"split"})
    rows = []
    for _, r in m.iterrows():
        fp = str(r.get("file_path",""))
        rows.append({"subject_id":r["subject_id"],"age":float(r["age"]),
                     "sex":str(r.get("sex","")),"split":str(r.get("split","")),
                     "cdr":float(r["cdr"]) if not pd.isna(r.get("cdr")) else None,
                     "mmse":float(r["mmse"]) if not pd.isna(r.get("mmse")) else None,
                     "file_exists":os.path.exists(fp),"file_path":fp})
    return rows

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_key_metrics())

@app.route("/api/charts/training")
def api_training():
    return jsonify(chart_training())

@app.route("/api/charts/scatter")
def api_scatter():
    return jsonify(chart_scatter())

@app.route("/api/charts/bag_sex")
def api_bag_sex():
    return jsonify(chart_bag_by_group("sex", ["#4f8ef7","#f472b6"]))

@app.route("/api/charts/bag_site")
def api_bag_site():
    return jsonify(chart_bag_by_group("site", ["#8b5cf6","#10b981","#f59e0b","#4f8ef7"]))

@app.route("/api/charts/mae_bar")
def api_mae_bar():
    return jsonify(chart_mae_bar())

@app.route("/api/charts/age_decade")
def api_age_decade():
    return jsonify(chart_age_decade_mae())

@app.route("/api/charts/cdr_bag")
def api_cdr_bag():
    return jsonify(chart_cdr_bag())

@app.route("/api/charts/mmse_bag")
def api_mmse_bag():
    return jsonify(chart_mmse_bag())

@app.route("/api/fairness_table")
def api_fairness():
    return jsonify(get_fairness_table())

@app.route("/api/clinical")
def api_clinical():
    return jsonify(get_oasis_clinical())

@app.route("/api/oasis_subjects")
def api_oasis_subjects():
    return jsonify(get_oasis_subjects())

# ── Live inference: file upload ────────────────────────────────────────────────
@app.route("/api/infer", methods=["POST"])
def api_infer():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f        = request.files["file"]
    sex      = request.form.get("sex", "unknown")
    true_age = request.form.get("true_age", None)
    cdr_in   = request.form.get("cdr", None)
    mmse_in  = request.form.get("mmse", None)
    fname    = f.filename.lower()
    data     = f.read()
    try:
        if fname.endswith(".pkl"):
            vol = preprocess_pkl(data); dataset = "IXI"
        elif fname.endswith(".nii.gz") or fname.endswith(".nii"):
            vol = preprocess_nii(data); dataset = "OASIS"
        elif fname.endswith(".hdr") or fname.endswith(".img"):
            # nibabel needs BOTH .hdr + .img files on disk together.
            # Extract subject ID from the uploaded filename and load from
            # the known OASIS disk path where both files already exist.
            orig_fname = f.filename  # preserve original case for subject ID
            sid_match = re.match(r"(OAS1_\d+_MR\d+)", orig_fname, re.IGNORECASE)
            if not sid_match:
                return jsonify({
                    "error": (
                        "Could not parse OASIS subject ID from filename. "
                        "Expected filename starting with OAS1_XXXX_MR1 "
                        "(e.g. OAS1_0027_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr). "
                        "Alternatively, use the Dementia Clinical tab subject selector "
                        "to run inference directly by subject ID."
                    )
                }), 400
            sid = sid_match.group(1).upper()
            # Look up the actual .hdr path from the OASIS manifest
            if not OASIS_CSV.exists():
                return jsonify({"error": "OASIS manifest not found on server."}), 404
            manifest = pd.read_csv(OASIS_CSV).rename(
                columns={"SUBJECT_ID": "subject_id", "FILE_PATH": "file_path",
                         "SEX": "sex", "CDR": "cdr", "MMSE": "mmse"})
            row = manifest[manifest.subject_id.str.upper() == sid]
            if row.empty:
                return jsonify({"error": f"Subject {sid} not found in OASIS manifest."}), 404
            hdr_disk = str(row.iloc[0]["file_path"])
            if not os.path.exists(hdr_disk):
                return jsonify({"error": f"File not found on disk: {hdr_disk}"}), 404
            # Auto-fill sex / CDR / MMSE from manifest if not provided by user
            r0 = row.iloc[0]
            if sex in ("unknown", ""):
                sex = str(r0.get("sex", "unknown"))
            if not cdr_in and not pd.isna(r0.get("cdr")):
                cdr_in = str(float(r0["cdr"]))
            if not mmse_in and not pd.isna(r0.get("mmse")):
                mmse_in = str(float(r0["mmse"]))
            vol = preprocess_hdr_path(hdr_disk)
            dataset = "OASIS"
        else:
            return jsonify({"error": "Unsupported format. Upload .pkl, .nii, .nii.gz, or .hdr"}), 400

        pred_age = _run_inference(vol, sex, dataset)
        bag = None; interpretation = ""; dementia_note = ""
        if true_age:
            ta  = float(true_age); bag = round(pred_age - ta, 2)
            interpretation = (
                "Brain age closely matches chronological age." if abs(bag) < 1
                else f"Brain appears {abs(bag):.1f} yrs {'OLDER' if bag>0 else 'YOUNGER'} than chronological age."
                     + (" Elevated BAG may indicate accelerated brain ageing." if bag>5 else "")
            )
            if cdr_in:
                cdr = float(cdr_in)
                cdr_label = {0.0:"Healthy",0.5:"Very Mild Dementia",1.0:"Mild Dementia",2.0:"Moderate Dementia"}.get(cdr,f"CDR {cdr}")
                expected  = {0.0:-1.3, 0.5:2.9, 1.0:4.2, 2.0:-0.9}.get(cdr, 0)
                dementia_note = (f"CDR {cdr} ({cdr_label}). Expected group BAG: {expected:+.1f}y. "
                                 f"This subject: {bag:+.1f}y.")

        fair = pd.read_csv(FAIR_CSV)
        group_mae = float(fair[fair.group_type == "Overall"].iloc[0]["MAE"])
        return jsonify({"predicted_age": round(pred_age,1), "true_age": float(true_age) if true_age else None,
                        "bag": bag, "interpretation": interpretation, "dementia_note": dementia_note,
                        "sex": sex, "dataset": dataset,
                        "cdr": float(cdr_in) if cdr_in else None,
                        "mmse": float(mmse_in) if mmse_in else None,
                        "model_mae": group_mae,
                        "confidence_note": f"Model uncertainty ~{group_mae:.1f} years"})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

# ── OASIS subject-ID inference (loads from E:\code\OASIS-1 directly) ──────────
@app.route("/api/infer_subject", methods=["POST"])
def api_infer_subject():
    body       = request.get_json() or {}
    subject_id = body.get("subject_id","").strip()
    if not subject_id:
        return jsonify({"error": "subject_id required"}), 400
    if not OASIS_CSV.exists():
        return jsonify({"error": "OASIS manifest not found"}), 404
    manifest = pd.read_csv(OASIS_CSV).rename(columns={
        "SUBJECT_ID":"subject_id","FILE_PATH":"file_path",
        "AGE":"age","SEX":"sex","CDR":"cdr","MMSE":"mmse"})
    row = manifest[manifest.subject_id == subject_id]
    if row.empty:
        return jsonify({"error": f"Subject {subject_id} not found"}), 404
    r = row.iloc[0]
    hdr_path = str(r["file_path"])
    if not os.path.exists(hdr_path):
        return jsonify({"error": f"File not found on disk: {hdr_path}"}), 404
    try:
        vol       = preprocess_hdr_path(hdr_path)
        sex_val   = str(r.get("sex", "unknown"))
        pred_age  = _run_inference(vol, sex_val, "OASIS", "WashU")
        ta        = float(r["age"])
        bag       = round(pred_age - ta, 2)
        cdr_val   = float(r["cdr"])  if not pd.isna(r.get("cdr"))  else None
        mmse_val  = float(r["mmse"]) if not pd.isna(r.get("mmse")) else None
        cdr_label = {0.0:"Healthy",0.5:"Very Mild",1.0:"Mild",2.0:"Moderate"}.get(cdr_val,"")
        expected  = {0.0:-1.3, 0.5:2.9, 1.0:4.2, 2.0:-0.9}.get(cdr_val, 0)
        interpretation = (f"Brain appears {bag:.1f} yrs OLDER than chronological age — elevated BAG." if bag>5
                          else f"Brain appears {abs(bag):.1f} yrs YOUNGER than chronological age." if bag<-5
                          else "BAG within normal range (±5 years).")
        fair = pd.read_csv(FAIR_CSV)
        group_mae = float(fair[fair.group_type == "Overall"].iloc[0]["MAE"])
        return jsonify({"subject_id": subject_id, "predicted_age": round(pred_age,1),
                        "true_age": ta, "bag": bag, "sex": sex_val,
                        "cdr": cdr_val, "cdr_label": cdr_label, "mmse": mmse_val,
                        "interpretation": interpretation,
                        "expected_bag_for_cdr": expected, "model_mae": group_mae})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("Brain-Age Fairness Audit Dashboard")
    print(f"   Device : {DEVICE}")
    print(f"   OASIS  : {'found' if OASIS_CSV.exists() else 'manifest not found'}")
    print("   Loading model ...")
    get_model()
    print("   Model loaded. -> http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
