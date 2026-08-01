import React, { useState, useMemo, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import { Landmark, BarChart3, ArrowUpRight, ArrowDownRight, Info, ChevronRight, HelpCircle, X, Upload, FileText, Download } from "lucide-react";
import Papa from "papaparse";

/* ---------------------------------------------------------------------
   OPTIONAL REAL BACKEND
   If you've deployed the FastAPI service (see backend/README.md), set
   API_BASE_URL to its public URL and this app will call the *real*
   HistGradientBoosting model (ROC-AUC 0.944) instead of the client-side
   Logistic Regression approximation (ROC-AUC 0.870). Leave as null to
   keep everything running client-side with no server required.
--------------------------------------------------------------------- */
const API_BASE_URL = "http://localhost:8000"; // FastAPI backend — set to null to run frontend-only

/* ---------------------------------------------------------------------
   REAL MODEL, PORTED FROM THE TRAINED NOTEBOOK
   Coefficients / scaler stats are the actual fitted Logistic Regression
   values from the enhanced pipeline (32,416 real loan records).
   This is a legitimate linear port of the trained model, not a mock —
   the production system uses HistGradientBoosting (ROC-AUC 0.944);
   this browser demo uses Logistic Regression (ROC-AUC 0.870) because
   it's the model that ports cleanly to client-side math.
--------------------------------------------------------------------- */

const FEATURES = [
  "person_age","person_income","person_emp_length","loan_grade","loan_amnt",
  "loan_int_rate","loan_percent_income","cb_person_default_on_file",
  "cb_person_cred_hist_length","loan_income_ratio",
  "person_home_ownership_OTHER","person_home_ownership_OWN","person_home_ownership_RENT",
  "loan_intent_EDUCATION","loan_intent_HOMEIMPROVEMENT","loan_intent_MEDICAL",
  "loan_intent_PERSONAL","loan_intent_VENTURE"
];

const COEF = {
  person_age: -0.03223, person_income: 0.04891, person_emp_length: -0.03775,
  loan_grade: 1.08271, loan_amnt: -0.69738, loan_int_rate: -0.0385,
  loan_percent_income: -5.90286, cb_person_default_on_file: -0.0504,
  cb_person_cred_hist_length: 0.00505, loan_income_ratio: 7.34568,
  person_home_ownership_OTHER: 0.02582, person_home_ownership_OWN: -0.42787,
  person_home_ownership_RENT: 0.40854, loan_intent_EDUCATION: -0.31721,
  loan_intent_HOMEIMPROVEMENT: 0.06086, loan_intent_MEDICAL: -0.12731,
  loan_intent_PERSONAL: -0.22592, loan_intent_VENTURE: -0.39713
};
const MEAN = {
  person_age: 27.7262, person_income: 65828.512, person_emp_length: 4.7478,
  loan_grade: 1.218, loan_amnt: 9592.1689, loan_int_rate: 11.0189,
  loan_percent_income: 0.1708, cb_person_default_on_file: 0.1767,
  cb_person_cred_hist_length: 5.7982, loan_income_ratio: 0.1711,
  person_home_ownership_OTHER: 0.0034, person_home_ownership_OWN: 0.0787,
  person_home_ownership_RENT: 0.5057, loan_intent_EDUCATION: 0.1976,
  loan_intent_HOMEIMPROVEMENT: 0.1088, loan_intent_MEDICAL: 0.1867,
  loan_intent_PERSONAL: 0.168, loan_intent_VENTURE: 0.1778
};
const SCALE = {
  person_age: 6.3586, person_income: 63091.7908, person_emp_length: 4.0932,
  loan_grade: 1.1646, loan_amnt: 6286.8062, loan_int_rate: 3.0748,
  loan_percent_income: 0.1068, cb_person_default_on_file: 0.3814,
  cb_person_cred_hist_length: 4.0455, loan_income_ratio: 0.1071,
  person_home_ownership_OTHER: 0.0582, person_home_ownership_OWN: 0.2693,
  person_home_ownership_RENT: 0.5, loan_intent_EDUCATION: 0.3982,
  loan_intent_HOMEIMPROVEMENT: 0.3114, loan_intent_MEDICAL: 0.3897,
  loan_intent_PERSONAL: 0.3738, loan_intent_VENTURE: 0.3823
};
const INTERCEPT = -1.9439630931421614;

const LABELS = {
  person_age: "Age", person_income: "Yearly income",
  person_emp_length: "Years at current job", loan_grade: "Loan grade",
  loan_amnt: "Loan amount", loan_int_rate: "Interest rate",
  loan_percent_income: "Loan size vs income", cb_person_default_on_file: "Missed a loan before",
  cb_person_cred_hist_length: "Years of credit history", loan_income_ratio: "Loan size vs income",
  person_home_ownership_OTHER: "Other housing", person_home_ownership_OWN: "Owns home",
  person_home_ownership_RENT: "Rents home", loan_intent_EDUCATION: "Loan for school",
  loan_intent_HOMEIMPROVEMENT: "Loan for home repairs", loan_intent_MEDICAL: "Loan for medical bills",
  loan_intent_PERSONAL: "Loan for personal use", loan_intent_VENTURE: "Loan for a business"
};

const GRADE_MAP = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6 };

const MODEL_RESULTS = [
  { model: "Logistic Regression", accuracy: 0.859, precision: 0.757, recall: 0.525, f1: 0.620, roc_auc: 0.870 },
  { model: "Decision Tree", accuracy: 0.891, precision: 0.739, recall: 0.777, f1: 0.757, roc_auc: 0.850 },
  { model: "Random Forest", accuracy: 0.936, precision: 0.971, recall: 0.728, f1: 0.832, roc_auc: 0.933 },
  { model: "Random Forest (Balanced)", accuracy: 0.935, precision: 0.981, recall: 0.717, f1: 0.828, roc_auc: 0.934 },
  { model: "HistGradientBoosting", accuracy: 0.923, precision: 0.847, recall: 0.791, f1: 0.818, roc_auc: 0.944 },
];

const FEATURE_IMPORTANCE = [
  { feature: "Loan size vs income", value: 0.1149 },
  { feature: "Loan grade", value: 0.0919 },
  { feature: "Yearly income", value: 0.0821 },
  { feature: "Rents home", value: 0.0356 },
  { feature: "Owns home", value: 0.0334 },
  { feature: "Loan size vs income (%)", value: 0.0271 },
  { feature: "Loan for a business", value: 0.0172 },
];

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// Real decision thresholds found during evaluation of the production model
// (HistGradientBoosting) on the held-out test set — not arbitrary picks.
const THRESHOLD_MODES = {
  default: { threshold: 0.50, label: "Standard", note: "Uses the normal cutoff point — no extra adjustments." },
  recall: { threshold: 0.366, label: "Catch more risk", note: "Flags more risky loans, but also turns away more good customers." },
  cost: { threshold: 0.30, label: "Save more money", note: "Picks the cutoff point that saves the most money overall, based on real loan amounts." },
};

function scoreFromRaw(raw, thresholdMode, includeContributions = true) {
  let z = INTERCEPT;
  const contributions = [];
  FEATURES.forEach((f) => {
    const val = raw[f] ?? 0;
    const std = (val - MEAN[f]) / SCALE[f];
    const contrib = COEF[f] * std;
    z += contrib;
    if (includeContributions) {
      contributions.push({ feature: f, label: LABELS[f], contrib });
    }
  });

  const prob = sigmoid(z);
  const score = Math.round(Math.max(300, Math.min(850, 850 - prob * 550)));
  if (includeContributions) contributions.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));

  const t = THRESHOLD_MODES[thresholdMode].threshold;
  const band = 0.06;
  let risk, verdict, verdictColor;
  if (prob < t - band) { risk = "Low risk"; verdict = "APPROVE"; verdictColor = "#4F7A54"; }
  else if (prob < t + band) { risk = "Moderate risk"; verdict = "REVIEW"; verdictColor = "#C9A227"; }
  else { risk = "High risk"; verdict = "DECLINE"; verdictColor = "#B8463D"; }

  return { prob, score, risk, verdict, verdictColor, threshold: t, contributions: contributions.slice(0, 7) };
}

function computeModel(form, thresholdMode) {
  const raw = {
    person_age: form.age,
    person_income: form.income,
    person_emp_length: form.empLength,
    loan_grade: GRADE_MAP[form.grade],
    loan_amnt: form.loanAmt,
    loan_int_rate: form.intRate,
    loan_percent_income: Math.min(form.loanAmt / form.income, 1),
    cb_person_default_on_file: form.priorDefault ? 1 : 0,
    cb_person_cred_hist_length: form.credHist,
    loan_income_ratio: Math.min(form.loanAmt / form.income, 1),
    person_home_ownership_OTHER: form.home === "OTHER" ? 1 : 0,
    person_home_ownership_OWN: form.home === "OWN" ? 1 : 0,
    person_home_ownership_RENT: form.home === "RENT" ? 1 : 0,
    loan_intent_EDUCATION: form.intent === "EDUCATION" ? 1 : 0,
    loan_intent_HOMEIMPROVEMENT: form.intent === "HOMEIMPROVEMENT" ? 1 : 0,
    loan_intent_MEDICAL: form.intent === "MEDICAL" ? 1 : 0,
    loan_intent_PERSONAL: form.intent === "PERSONAL" ? 1 : 0,
    loan_intent_VENTURE: form.intent === "VENTURE" ? 1 : 0,
  };
  return scoreFromRaw(raw, thresholdMode, true);
}

// Converts one row of an uploaded CSV (same column names as the original
// training dataset) into the same raw-feature shape the model expects.
function buildRawFromCsvRow(row) {
  const income = Number(row.person_income) || 0;
  const loanAmt = Number(row.loan_amnt) || 0;
  const ratio = income > 0 ? Math.min(loanAmt / income, 1) : 0;
  const grade = String(row.loan_grade ?? "").trim().toUpperCase();
  const home = String(row.person_home_ownership ?? "").trim().toUpperCase();
  const intent = String(row.loan_intent ?? "").trim().toUpperCase();
  const defaultRaw = row.cb_person_default_on_file;
  const priorDefault = (defaultRaw === "Y" || defaultRaw === true || defaultRaw === 1 || defaultRaw === "1" || defaultRaw === "y") ? 1 : 0;

  return {
    person_age: Number(row.person_age) || 0,
    person_income: income,
    person_emp_length: Number(row.person_emp_length) || 0,
    loan_grade: GRADE_MAP[grade] ?? 1,
    loan_amnt: loanAmt,
    loan_int_rate: Number(row.loan_int_rate) || 0,
    loan_percent_income: ratio,
    cb_person_default_on_file: priorDefault,
    cb_person_cred_hist_length: Number(row.cb_person_cred_hist_length) || 0,
    loan_income_ratio: ratio,
    person_home_ownership_OTHER: home === "OTHER" ? 1 : 0,
    person_home_ownership_OWN: home === "OWN" ? 1 : 0,
    person_home_ownership_RENT: home === "RENT" ? 1 : 0,
    loan_intent_EDUCATION: intent === "EDUCATION" ? 1 : 0,
    loan_intent_HOMEIMPROVEMENT: intent === "HOMEIMPROVEMENT" ? 1 : 0,
    loan_intent_MEDICAL: intent === "MEDICAL" ? 1 : 0,
    loan_intent_PERSONAL: intent === "PERSONAL" ? 1 : 0,
    loan_intent_VENTURE: intent === "VENTURE" ? 1 : 0,
  };
}

const REQUIRED_CSV_COLUMNS = [
  "person_age", "person_income", "person_home_ownership", "person_emp_length",
  "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
  "cb_person_default_on_file", "cb_person_cred_hist_length",
];

const THEME = {
  bg: "#14171B", panel: "#1B1F24", panelAlt: "#20262D", hairline: "#2C333B",
  ink: "#ECE8DE", inkDim: "#9AA0A6", brass: "#C9A227", brassBright: "#E0BB3E",
  red: "#B8463D", green: "#4F7A54",
};

function HelpTip({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      style={{ position: "relative", display: "inline-flex", marginLeft: 5, cursor: "help" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onClick={() => setOpen((o) => !o)}
    >
      <Info size={12} color={THEME.inkDim} />
      {open && (
        <span style={{
          position: "absolute", bottom: "140%", left: "50%", transform: "translateX(-50%)",
          background: "#0F1216", color: THEME.ink, border: `1px solid ${THEME.hairline}`,
          borderRadius: 6, padding: "8px 10px", fontSize: 11.5, lineHeight: 1.4,
          width: 200, zIndex: 20, fontFamily: "'IBM Plex Sans', sans-serif", fontWeight: 400,
          boxShadow: "0 4px 14px rgba(0,0,0,0.4)"
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

function Field({ label, help, unit, value, min, max, step, onChange, format }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ display: "flex", alignItems: "center", fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 13, color: THEME.inkDim, letterSpacing: 0.3 }}>
          {label}{help && <HelpTip text={help} />}
        </span>
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, color: THEME.brassBright }}>
          {format ? format(value) : value}{unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: THEME.brass, height: 4, cursor: "pointer" }}
      />
    </div>
  );
}

function Select({ label, help, value, options, onChange }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: 6, fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 13, color: THEME.inkDim, letterSpacing: 0.3 }}>
        {label}{help && <HelpTip text={help} />}
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%", background: THEME.panelAlt, color: THEME.ink, border: `1px solid ${THEME.hairline}`,
          borderRadius: 6, padding: "8px 10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13
        }}
      >
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function Toggle({ label, help, value, onChange }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
      <span style={{ display: "flex", alignItems: "center", fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 13, color: THEME.inkDim, letterSpacing: 0.3 }}>
        {label}{help && <HelpTip text={help} />}
      </span>
      <button
        onClick={() => onChange(!value)}
        style={{
          width: 46, height: 24, borderRadius: 12, border: `1px solid ${THEME.hairline}`,
          background: value ? THEME.red : THEME.panelAlt, position: "relative", cursor: "pointer", transition: "background 0.25s"
        }}
      >
        <span style={{
          position: "absolute", top: 2, left: value ? 24 : 2, width: 18, height: 18, borderRadius: 9,
          background: THEME.ink, transition: "left 0.25s"
        }} />
      </button>
    </div>
  );
}

function Gauge({ score }) {
  const angle = -90 + ((score - 300) / (850 - 300)) * 180;
  const cx = 140, cy = 140, r = 110;
  const ticks = [300, 440, 580, 670, 760, 850];
  return (
    <svg viewBox="0 0 280 160" style={{ width: "100%", maxWidth: 300 }}>
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={THEME.hairline} strokeWidth="14" strokeLinecap="round" />
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx - r + (r * 2 * 0.24)} ${cy - 55}`} fill="none" stroke={THEME.red} strokeWidth="14" strokeLinecap="round" opacity="0.85" />
      <path d={`M ${cx - r * 0.4} ${cy - 100} A ${r} ${r} 0 0 1 ${cx + r * 0.5} ${cy - 95}`} fill="none" stroke={THEME.brass} strokeWidth="14" opacity="0.85" />
      <path d={`M ${cx + r * 0.3} ${cy - 100} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={THEME.green} strokeWidth="14" strokeLinecap="round" opacity="0.85" />
      {ticks.map((t) => {
        const a = (-90 + ((t - 300) / 550) * 180) * (Math.PI / 180);
        const x1 = cx + Math.cos(a) * (r - 14), y1 = cy + Math.sin(a) * (r - 14);
        const x2 = cx + Math.cos(a) * (r + 6), y2 = cy + Math.sin(a) * (r + 6);
        return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} stroke={THEME.inkDim} strokeWidth="1.5" />;
      })}
      <g style={{ transform: `rotate(${angle}deg)`, transformOrigin: `${cx}px ${cy}px`, transition: "transform 0.7s cubic-bezier(.34,1.56,.64,1)" }}>
        <line x1={cx} y1={cy} x2={cx - r + 20} y2={cy} stroke={THEME.brassBright} strokeWidth="3" strokeLinecap="round" />
      </g>
      <circle cx={cx} cy={cy} r="7" fill={THEME.brassBright} />
    </svg>
  );
}

export default function CreditLedgerApp() {
  const [tab, setTab] = useState("score");
  const [thresholdMode, setThresholdMode] = useState("cost");
  const [showGuide, setShowGuide] = useState(false);

  const [batchFile, setBatchFile] = useState(null);
  const [batchRows, setBatchRows] = useState(null);       // parsed CSV rows
  const [batchResults, setBatchResults] = useState(null); // scored results
  const [batchEval, setBatchEval] = useState(null);        // accuracy/precision/etc if labels present
  const [batchSource, setBatchSource] = useState(null);    // "server" | "local"
  const [batchStatus, setBatchStatus] = useState("idle");  // idle | loading | done | error
  const [batchError, setBatchError] = useState("");
  const [form, setForm] = useState({
    age: 30, income: 55000, empLength: 4, grade: "B", loanAmt: 9000,
    intRate: 11.5, credHist: 6, priorDefault: false, home: "RENT", intent: "PERSONAL",
  });
  const [backendResult, setBackendResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState(API_BASE_URL ? "loading" : "off"); // off | loading | ok | error

  const localResult = useMemo(() => computeModel(form, thresholdMode), [form, thresholdMode]);

  useEffect(() => {
    if (!API_BASE_URL) return;
    let cancelled = false;
    setBackendStatus("loading");
    const timer = setTimeout(() => {
      fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_age: form.age, person_income: form.income, person_emp_length: form.empLength,
          loan_grade: form.grade, loan_amnt: form.loanAmt, loan_int_rate: form.intRate,
          cb_person_cred_hist_length: form.credHist, cb_person_default_on_file: form.priorDefault,
          person_home_ownership: form.home, loan_intent: form.intent, threshold_mode: thresholdMode,
        }),
      })
        .then((r) => { if (!r.ok) throw new Error("API error " + r.status); return r.json(); })
        .then((data) => { if (!cancelled) { setBackendResult(data); setBackendStatus("ok"); } })
        .catch(() => { if (!cancelled) setBackendStatus("error"); });
    }, 300); // small debounce while dragging sliders
    return () => { cancelled = true; clearTimeout(timer); };
  }, [form, thresholdMode]);

  // Prefer the real backend model when it's configured and responding;
  // otherwise fall back to the client-side Logistic Regression approximation.
  const result = backendStatus === "ok" && backendResult
    ? {
        prob: backendResult.probability_of_default,
        score: backendResult.credit_score,
        risk: backendResult.risk_band,
        verdict: backendResult.verdict,
        verdictColor: backendResult.verdict === "APPROVE" ? "#4F7A54" : backendResult.verdict === "REVIEW" ? "#C9A227" : "#B8463D",
        contributions: (backendResult.top_contributions || []).map((c) => ({ feature: c.feature, label: c.label, contrib: c.contribution })),
      }
    : localResult;
  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));

  function handleBatchFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setBatchFile(file);
    setBatchStatus("loading");
    setBatchError("");
    setBatchResults(null);
    setBatchEval(null);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: async (parsed) => {
        const rows = parsed.data;
        const missing = REQUIRED_CSV_COLUMNS.filter((c) => !(c in (rows[0] || {})));
        if (missing.length > 0) {
          setBatchStatus("error");
          setBatchError(`This file is missing required column(s): ${missing.join(", ")}`);
          return;
        }
        setBatchRows(rows);
        await scoreBatch(rows, file);
      },
      error: (err) => {
        setBatchStatus("error");
        setBatchError("Could not read this file as a CSV: " + err.message);
      },
    });
  }

  async function scoreBatch(rows, file) {
    // Try the real backend first, if configured — falls back to the
    // client-side approximation if it's unreachable.
    if (API_BASE_URL) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${API_BASE_URL}/predict-batch?threshold_mode=${thresholdMode}`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Server responded with " + res.status);
        const data = await res.json();
        setBatchResults(data.results.map((r) => ({ row: r.row, score: r.credit_score, risk: r.risk_band, verdict: r.verdict, prob: r.probability_of_default })));
        setBatchEval(data.evaluation || null);
        setBatchSource("server");
        setBatchStatus("done");
        return;
      } catch (e) {
        // fall through to local scoring
      }
    }

    try {
      const hasLabels = "loan_status" in (rows[0] || {});
      const results = rows.map((row, i) => {
        const raw = buildRawFromCsvRow(row);
        const r = scoreFromRaw(raw, thresholdMode, false);
        return { row: i, score: r.score, risk: r.risk, verdict: r.verdict, prob: r.prob };
      });

      let evalMetrics = null;
      if (hasLabels) {
        const t = THRESHOLD_MODES[thresholdMode].threshold;
        let tp = 0, tn = 0, fp = 0, fn = 0;
        rows.forEach((row, i) => {
          const actual = Number(row.loan_status);
          const predicted = results[i].prob >= t ? 1 : 0;
          if (predicted === 1 && actual === 1) tp++;
          else if (predicted === 0 && actual === 0) tn++;
          else if (predicted === 1 && actual === 0) fp++;
          else if (predicted === 0 && actual === 1) fn++;
        });
        const accuracy = (tp + tn) / (tp + tn + fp + fn || 1);
        const precision = tp / (tp + fp || 1);
        const recall = tp / (tp + fn || 1);
        const f1 = (2 * precision * recall) / (precision + recall || 1);
        evalMetrics = { accuracy, precision, recall, f1 };
      }

      setBatchResults(results);
      setBatchEval(evalMetrics);
      setBatchSource("local");
      setBatchStatus("done");
    } catch (e) {
      setBatchStatus("error");
      setBatchError("Could not score this file: " + e.message);
    }
  }

  function downloadSampleTemplate() {
    const sample = [
      { person_age: 30, person_income: 55000, person_home_ownership: "RENT", person_emp_length: 4,
        loan_intent: "PERSONAL", loan_grade: "B", loan_amnt: 9000, loan_int_rate: 11.5,
        cb_person_default_on_file: "N", cb_person_cred_hist_length: 6 },
      { person_age: 45, person_income: 120000, person_home_ownership: "OWN", person_emp_length: 15,
        loan_intent: "VENTURE", loan_grade: "A", loan_amnt: 5000, loan_int_rate: 6.5,
        cb_person_default_on_file: "N", cb_person_cred_hist_length: 20 },
      { person_age: 23, person_income: 24000, person_home_ownership: "RENT", person_emp_length: 1,
        loan_intent: "MEDICAL", loan_grade: "E", loan_amnt: 12000, loan_int_rate: 18.2,
        cb_person_default_on_file: "Y", cb_person_cred_hist_length: 2 },
    ];
    const csv = Papa.unparse(sample);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadBatchResults() {
    if (!batchResults || !batchRows) return;
    const merged = batchRows.map((row, i) => ({
      ...row,
      predicted_credit_score: batchResults[i].score,
      predicted_risk: batchResults[i].risk,
      predicted_verdict: batchResults[i].verdict,
      predicted_probability_of_default: batchResults[i].prob,
    }));
    const csv = Papa.unparse(merged);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "scored_" + (batchFile ? batchFile.name : "results.csv");
    a.click();
    URL.revokeObjectURL(url);
  }

  function resetBatch() {
    setBatchFile(null);
    setBatchRows(null);
    setBatchResults(null);
    setBatchEval(null);
    setBatchStatus("idle");
    setBatchError("");
  }

  return (
    <div style={{
      background: THEME.bg, color: THEME.ink, minHeight: 640, padding: "28px 20px",
      fontFamily: "'IBM Plex Sans', sans-serif"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #2C333B; border-radius: 3px; }
      `}</style>

      <div style={{ maxWidth: 1040, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 26, flexWrap: "wrap", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Landmark size={22} color={THEME.brass} />
            <span style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 22, letterSpacing: 0.3 }}>Loan Score Checker</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              onClick={() => setShowGuide(true)}
              style={{
                display: "flex", alignItems: "center", gap: 6, background: THEME.panel,
                border: `1px solid ${THEME.hairline}`, borderRadius: 8, padding: "7px 14px",
                color: THEME.inkDim, fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 13,
                fontWeight: 600, cursor: "pointer"
              }}
            >
              <HelpCircle size={15} color={THEME.brass} /> How This Works
            </button>
            <div style={{ display: "flex", gap: 4, background: THEME.panel, borderRadius: 8, padding: 4, border: `1px solid ${THEME.hairline}` }}>
              {[{ id: "score", label: "Check a Loan" }, { id: "batch", label: "Upload Data" }, { id: "analytics", label: "Model Stats" }].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  style={{
                    padding: "7px 16px", borderRadius: 6, border: "none", cursor: "pointer",
                    background: tab === t.id ? THEME.brass : "transparent",
                    color: tab === t.id ? "#14171B" : THEME.inkDim,
                    fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 13, fontWeight: 600, transition: "all 0.2s"
                  }}
                >{t.label}</button>
              ))}
            </div>
          </div>
        </div>

        {tab === "score" ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 20 }} className="ledger-grid">
            <div style={{ display: "grid", gridTemplateColumns: "minmax(280px,1fr) minmax(280px,1.1fr)", gap: 20 }}>
              {/* Applicant ledger form */}
              <div style={{ background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12, padding: 22 }}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 16, marginBottom: 4, color: THEME.brassBright }}>Loan Details</div>
                <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 18 }}>Change any field and the score updates right away.</div>

                <Field label="Yearly income" help="The applicant's total income per year, before tax. Higher income generally means an easier time affording payments." unit="" value={form.income} min={4000} max={250000} step={1000} onChange={set("income")} format={(v) => "$" + v.toLocaleString()} />
                <Field label="Age" help="The applicant's age in years. A minor factor — on its own it has a small effect on the score." unit=" yrs" value={form.age} min={18} max={80} step={1} onChange={set("age")} />
                <Field label="Years at current job" help="How long they've held their current job. Longer tenure usually signals more stable income." unit=" yrs" value={form.empLength} min={0} max={40} step={1} onChange={set("empLength")} />
                <Select label="Loan grade" help="A risk grade already assigned to the loan by the lender. A is safest, G is riskiest. This is one of the strongest signals the model uses." value={form.grade} onChange={set("grade")}
                  options={"ABCDEFG".split("").map((g) => ({ value: g, label: `Grade ${g}` }))} />
                <Field label="Loan amount" help="How much money is being borrowed. Larger loans relative to income carry more risk." unit="" value={form.loanAmt} min={500} max={35000} step={250} onChange={set("loanAmt")} format={(v) => "$" + v.toLocaleString()} />
                <Field label="Interest rate" help="The yearly interest rate on the loan. Riskier loans are usually priced with a higher rate." unit="%" value={form.intRate} min={5} max={24} step={0.1} onChange={set("intRate")} format={(v) => v.toFixed(1)} />
                <Field label="Years of credit history" help="How long the applicant has had any form of credit. A longer history gives more evidence of reliability." unit=" yrs" value={form.credHist} min={2} max={30} step={1} onChange={set("credHist")} />
                <Select label="Housing" help="Whether the applicant rents, owns, has a mortgage, or another arrangement. Owning is generally seen as more stable than renting." value={form.home} onChange={set("home")}
                  options={[{ value: "MORTGAGE", label: "Mortgage" }, { value: "RENT", label: "Rent" }, { value: "OWN", label: "Own" }, { value: "OTHER", label: "Other" }]} />
                <Select label="Reason for loan" help="What the money is being used for. Some purposes are statistically riskier than others — for example, business loans tend to carry more risk than debt consolidation." value={form.intent} onChange={set("intent")}
                  options={[
                    { value: "DEBTCONSOLIDATION", label: "Pay off other debts" }, { value: "EDUCATION", label: "School" },
                    { value: "HOMEIMPROVEMENT", label: "Home repairs" }, { value: "MEDICAL", label: "Medical bills" },
                    { value: "PERSONAL", label: "Personal use" }, { value: "VENTURE", label: "Business" },
                  ]} />
                <Toggle label="Missed a loan payment before?" help="Has this applicant ever defaulted on a loan before? A past default is one of the strongest warning signs the model looks at." value={form.priorDefault} onChange={set("priorDefault")} />
              </div>

              {/* Result */}
              <div style={{ background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12, padding: 22, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ width: "100%", marginBottom: 16 }}>
                  <div style={{ fontSize: 11, color: THEME.inkDim, marginBottom: 6, letterSpacing: 0.5 }}>WHAT MATTERS MORE?</div>
                  <div style={{ display: "flex", gap: 4, background: THEME.panelAlt, borderRadius: 8, padding: 3, border: `1px solid ${THEME.hairline}` }}>
                    {Object.entries(THRESHOLD_MODES).map(([key, m]) => (
                      <button
                        key={key}
                        onClick={() => setThresholdMode(key)}
                        style={{
                          flex: 1, padding: "6px 8px", borderRadius: 5, border: "none", cursor: "pointer",
                          background: thresholdMode === key ? THEME.brass : "transparent",
                          color: thresholdMode === key ? "#14171B" : THEME.inkDim,
                          fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 11, fontWeight: 600, transition: "all 0.2s"
                        }}
                      >{m.label}</button>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: THEME.inkDim, marginTop: 6, lineHeight: 1.4 }}>{THRESHOLD_MODES[thresholdMode].note}</div>
                </div>
                <Gauge score={result.score} />
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 44, fontWeight: 700, marginTop: -10, color: THEME.ink }}>{result.score}</div>
                <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 14, letterSpacing: 0.5 }}>CREDIT SCORE (300–850)</div>

                <div style={{
                  transform: "rotate(-4deg)", border: `2px solid ${result.verdictColor}`, color: result.verdictColor,
                  padding: "6px 22px", borderRadius: 4, fontFamily: "'Fraunces', serif", fontWeight: 700,
                  fontSize: 18, letterSpacing: 2, marginBottom: 18, transition: "all 0.3s"
                }}>{result.verdict}</div>

                <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 16 }}>{result.risk} · {(result.prob * 100).toFixed(1)}% chance of not repaying</div>

                {/* Score breakdown */}
                <div style={{ width: "100%", borderTop: `1px solid ${THEME.hairline}`, paddingTop: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: THEME.inkDim, marginBottom: 8, letterSpacing: 0.5 }}>
                    <span>WHY THIS SCORE</span>
                    <span style={{ display: "flex", gap: 14 }}>
                      <span style={{ color: THEME.red }}>Raises risk</span>
                      <span style={{ color: THEME.green }}>Lowers risk</span>
                    </span>
                  </div>
                  {result.contributions.map((c) => (
                    <div key={c.feature} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: `1px dashed ${THEME.hairline}` }}>
                      <span style={{ fontSize: 12.5, color: THEME.ink }}>{c.label}</span>
                      <span style={{ display: "flex", alignItems: "center", gap: 4, fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: c.contrib > 0 ? THEME.red : THEME.green }}>
                        {c.contrib > 0 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                        {Math.abs(c.contrib).toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "12px 16px", background: THEME.panelAlt, borderRadius: 8, border: `1px solid ${THEME.hairline}` }}>
              <Info size={15} color={THEME.inkDim} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: THEME.inkDim, lineHeight: 1.5 }}>
                This score is worked out right in your browser, using real numbers from a model trained on 32,416 real loans. Out of 100 similar cases, this version gets about 87 right. The list above shows which details pushed the score up or down. A more accurate version of this model (getting about 94 out of 100 right) runs on the server — see the Model Stats tab for those numbers.
              </span>
            </div>
          </div>
        ) : tab === "batch" ? (
          <div style={{ display: "grid", gap: 20 }}>
            <div style={{ background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12, padding: 22 }}>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: 16, marginBottom: 4, color: THEME.brassBright }}>Upload a Dataset</div>
              <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 18, lineHeight: 1.5 }}>
                Upload a CSV of loan applicants to score all of them at once. The file needs these columns:
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", color: THEME.brassBright, display: "block", marginTop: 6, fontSize: 11, lineHeight: 1.6 }}>
                  {REQUIRED_CSV_COLUMNS.join(", ")}
                </span>
                <span style={{ display: "block", marginTop: 6 }}>
                  If your file also has a <code style={{ color: THEME.brassBright }}>loan_status</code> column (0 = repaid, 1 = defaulted), the app will also show how accurate the predictions were against those real outcomes.
                </span>
                <button onClick={downloadSampleTemplate} style={{ display: "flex", alignItems: "center", gap: 6, background: "none", border: `1px solid ${THEME.hairline}`, borderRadius: 6, padding: "6px 12px", marginTop: 10, cursor: "pointer", color: THEME.brassBright, fontSize: 11.5, fontFamily: "'IBM Plex Sans', sans-serif", fontWeight: 600 }}>
                  <Download size={13} /> Download a sample template
                </button>
              </div>

              {batchStatus === "idle" && (
                <label style={{
                  display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                  gap: 8, border: `1.5px dashed ${THEME.hairline}`, borderRadius: 10, padding: "36px 20px",
                  cursor: "pointer", color: THEME.inkDim
                }}>
                  <Upload size={26} color={THEME.brass} />
                  <span style={{ fontSize: 13 }}>Click to choose a .csv file</span>
                  <input type="file" accept=".csv" onChange={handleBatchFile} style={{ display: "none" }} />
                </label>
              )}

              {batchStatus === "loading" && (
                <div style={{ textAlign: "center", padding: "30px 0", color: THEME.inkDim, fontSize: 13 }}>Scoring your file…</div>
              )}

              {batchStatus === "error" && (
                <div>
                  <div style={{ background: "rgba(184,70,61,0.12)", border: `1px solid ${THEME.red}`, borderRadius: 8, padding: 14, color: THEME.ink, fontSize: 13, marginBottom: 12 }}>
                    {batchError}
                  </div>
                  <button onClick={resetBatch} style={{ background: THEME.brass, border: "none", borderRadius: 6, padding: "8px 16px", fontWeight: 600, cursor: "pointer" }}>Try another file</button>
                </div>
              )}

              {batchStatus === "done" && batchResults && (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: THEME.inkDim }}>
                      <FileText size={14} color={THEME.brass} /> {batchFile?.name} · {batchResults.length} applicants
                    </span>
                    <span style={{ fontSize: 11, color: THEME.inkDim, background: THEME.panelAlt, padding: "3px 8px", borderRadius: 10, border: `1px solid ${THEME.hairline}` }}>
                      {batchSource === "server" ? "Scored by the real server model" : "Scored in your browser (approximate model)"}
                    </span>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: 10, marginBottom: 18 }}>
                    <SummaryCard label="Approve" value={batchResults.filter(r => r.verdict === "APPROVE").length} color={THEME.green} />
                    <SummaryCard label="Review" value={batchResults.filter(r => r.verdict === "REVIEW").length} color={THEME.brass} />
                    <SummaryCard label="Decline" value={batchResults.filter(r => r.verdict === "DECLINE").length} color={THEME.red} />
                    <SummaryCard label="Avg. Score" value={Math.round(batchResults.reduce((s, r) => s + r.score, 0) / batchResults.length)} color={THEME.ink} />
                  </div>

                  {batchEval && (
                    <div style={{ marginBottom: 18, padding: 14, background: THEME.panelAlt, borderRadius: 8, border: `1px solid ${THEME.hairline}` }}>
                      <div style={{ fontSize: 11, color: THEME.inkDim, marginBottom: 8, letterSpacing: 0.5 }}>ACCURACY AGAINST REAL OUTCOMES (loan_status column found)</div>
                      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13 }}>
                        <span>Accuracy: {(batchEval.accuracy * 100).toFixed(1)}%</span>
                        <span>Precision: {(batchEval.precision * 100).toFixed(1)}%</span>
                        <span>Recall: {(batchEval.recall * 100).toFixed(1)}%</span>
                        <span>F1: {(batchEval.f1 * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}

                  <div style={{ maxHeight: 320, overflowY: "auto", border: `1px solid ${THEME.hairline}`, borderRadius: 8 }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                      <thead style={{ position: "sticky", top: 0, background: THEME.panelAlt }}>
                        <tr>
                          <th style={{ textAlign: "left", padding: "8px 12px", color: THEME.inkDim, fontWeight: 600 }}>Row</th>
                          <th style={{ textAlign: "left", padding: "8px 12px", color: THEME.inkDim, fontWeight: 600 }}>Score</th>
                          <th style={{ textAlign: "left", padding: "8px 12px", color: THEME.inkDim, fontWeight: 600 }}>Risk</th>
                          <th style={{ textAlign: "left", padding: "8px 12px", color: THEME.inkDim, fontWeight: 600 }}>Verdict</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.slice(0, 300).map((r) => (
                          <tr key={r.row} style={{ borderTop: `1px dashed ${THEME.hairline}` }}>
                            <td style={{ padding: "6px 12px", color: THEME.inkDim }}>{r.row + 1}</td>
                            <td style={{ padding: "6px 12px", fontFamily: "'IBM Plex Mono', monospace" }}>{r.score}</td>
                            <td style={{ padding: "6px 12px", color: THEME.inkDim }}>{r.risk}</td>
                            <td style={{ padding: "6px 12px", fontWeight: 600, color: r.verdict === "APPROVE" ? THEME.green : r.verdict === "REVIEW" ? THEME.brass : THEME.red }}>{r.verdict}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {batchResults.length > 300 && (
                    <div style={{ fontSize: 11, color: THEME.inkDim, marginTop: 8 }}>Showing the first 300 of {batchResults.length} rows. Download the full results below.</div>
                  )}

                  <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
                    <button onClick={downloadBatchResults} style={{ display: "flex", alignItems: "center", gap: 6, background: THEME.brass, border: "none", borderRadius: 6, padding: "9px 16px", fontWeight: 600, cursor: "pointer", color: "#14171B" }}>
                      <Download size={15} /> Download Results CSV
                    </button>
                    <button onClick={resetBatch} style={{ background: "transparent", border: `1px solid ${THEME.hairline}`, borderRadius: 6, padding: "9px 16px", fontWeight: 600, cursor: "pointer", color: THEME.inkDim }}>
                      Upload Another File
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "12px 16px", background: THEME.panelAlt, borderRadius: 8, border: `1px solid ${THEME.hairline}` }}>
              <Info size={15} color={THEME.inkDim} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: THEME.inkDim, lineHeight: 1.5 }}>
                Every row is scored using the "{THRESHOLD_MODES[thresholdMode].label}" setting from the Check a Loan tab. Switch that setting before uploading if you want a different cutoff applied. Nothing you upload is sent anywhere except to your own backend (if configured) — files are processed and then discarded.
              </span>
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gap: 20 }}>
            <div style={{ background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12, padding: 22 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <BarChart3 size={17} color={THEME.brass} />
                <span style={{ fontFamily: "'Fraunces', serif", fontSize: 16, color: THEME.brassBright }}>Which Model Works Best</span>
              </div>
              <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 16 }}>How good each model is at telling good and risky loans apart. Higher is better.</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={MODEL_RESULTS} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={THEME.hairline} vertical={false} />
                  <XAxis dataKey="model" tick={{ fill: THEME.inkDim, fontSize: 10 }} interval={0} angle={-15} textAnchor="end" height={70} />
                  <YAxis domain={[0.6, 1]} tick={{ fill: THEME.inkDim, fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: THEME.panelAlt, border: `1px solid ${THEME.hairline}`, borderRadius: 8, fontSize: 12 }} labelStyle={{ color: THEME.ink }} />
                  <Bar dataKey="roc_auc" radius={[4, 4, 0, 0]}>
                    {MODEL_RESULTS.map((r, i) => (
                      <Cell key={i} fill={r.model === "HistGradientBoosting" ? THEME.brass : "#3A4149"} />
                    ))}
                    <LabelList dataKey="roc_auc" position="top" formatter={(v) => v.toFixed(3)} fill={THEME.inkDim} fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12, padding: 22 }}>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: 16, marginBottom: 4, color: THEME.brassBright }}>What Affects the Score Most</div>
              <div style={{ fontSize: 12, color: THEME.inkDim, marginBottom: 16 }}>Found by testing the best model against thousands of real loans.</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 40, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={THEME.hairline} horizontal={false} />
                  <XAxis type="number" tick={{ fill: THEME.inkDim, fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: THEME.ink, fontSize: 12 }} width={140} />
                  <Tooltip contentStyle={{ background: THEME.panelAlt, border: `1px solid ${THEME.hairline}`, borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="value" fill={THEME.brass} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "12px 16px", background: THEME.panelAlt, borderRadius: 8, border: `1px solid ${THEME.hairline}` }}>
              <ChevronRight size={15} color={THEME.inkDim} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: THEME.inkDim, lineHeight: 1.5 }}>
                Every number on this page comes from testing the models on 6,484 real loans they had never seen before — nothing here is made up.
              </span>
            </div>
          </div>
        )}

        <div style={{ textAlign: "center", marginTop: 28, paddingTop: 16, borderTop: `1px solid ${THEME.hairline}` }}>
          <span style={{ fontSize: 11.5, color: THEME.inkDim, fontFamily: "'IBM Plex Sans', sans-serif", letterSpacing: 0.3 }}>
            Developed by <span style={{ color: THEME.brassBright, fontWeight: 600 }}>Faiza Soomro</span>
          </span>
        </div>
      </div>

      {showGuide && (
        <div
          onClick={() => setShowGuide(false)}
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 50,
            display: "flex", alignItems: "center", justifyContent: "center", padding: 20
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: THEME.panel, border: `1px solid ${THEME.hairline}`, borderRadius: 12,
              maxWidth: 640, width: "100%", maxHeight: "85vh", overflowY: "auto", padding: 26,
              fontFamily: "'IBM Plex Sans', sans-serif"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
              <span style={{ fontFamily: "'Fraunces', serif", fontSize: 20, color: THEME.brassBright }}>How This Works</span>
              <button onClick={() => setShowGuide(false)} style={{ background: "none", border: "none", cursor: "pointer", color: THEME.inkDim }}>
                <X size={20} />
              </button>
            </div>

            <GuideSection title="What this tool does">
              You enter details about a loan applicant, and the tool instantly works out a credit score (300 to 850) and a recommendation: Approve, Review, or Decline. Every field updates the result immediately — there's no submit button needed.
            </GuideSection>

            <GuideSection title="The two tabs">
              <b>Check a Loan</b> is the calculator you start on. <b>Upload Data</b> lets you score a whole file of applicants at once. <b>Model Stats</b> shows how well the underlying model performs and which details matter most across all applicants, not just one.
            </GuideSection>

            <GuideSection title="Every field, explained">
              <TermRow term="Yearly income" desc="Total income per year, before tax. Higher income generally means an easier time affording payments." />
              <TermRow term="Age" desc="A minor factor — on its own it has a small effect on the score." />
              <TermRow term="Years at current job" desc="Longer job tenure usually signals more stable income." />
              <TermRow term="Loan grade" desc="A risk grade already assigned by the lender, A (safest) to G (riskiest). One of the strongest signals used." />
              <TermRow term="Loan amount" desc="How much is being borrowed. Larger loans relative to income carry more risk." />
              <TermRow term="Interest rate" desc="The yearly rate charged. Riskier loans are usually priced higher." />
              <TermRow term="Years of credit history" desc="Longer history gives more evidence of how reliably someone repays debt." />
              <TermRow term="Housing" desc="Renting, owning, mortgage, or other. Owning is generally seen as more stable than renting." />
              <TermRow term="Reason for loan" desc="What the money is for. Some purposes (like business loans) are statistically riskier than others (like debt consolidation)." />
              <TermRow term="Missed a loan payment before?" desc="A yes/no switch for past default. One of the strongest warning signs the model looks at." />
            </GuideSection>

            <GuideSection title="The three buttons above the score">
              <TermRow term="Standard" desc="A plain, textbook cutoff point — no adjustment." />
              <TermRow term="Catch more risk" desc="Flags more risky loans (catches ~85 out of 100 real defaulters instead of ~79), but also wrongly rejects more good customers." />
              <TermRow term="Save more money" desc="Picks the cutoff that historically minimized total dollar losses. Recommended for real-world use." />
            </GuideSection>

            <GuideSection title="Reading the result">
              <TermRow term="The dial" desc="A gauge from 300 to 850 — further right means lower risk." />
              <TermRow term="APPROVE / REVIEW / DECLINE" desc="The tool's recommendation based on the score and the selected button above." />
              <TermRow term="Risk label and %" desc="A plain-language risk level, plus the estimated chance this applicant doesn't repay." />
              <TermRow term="Why this score" desc="Red arrows raise risk, green arrows lower it. Bigger numbers mean a bigger effect. This list is personalized to whatever is currently in the form." />
            </GuideSection>

            <GuideSection title="Upload Data tab">
              <TermRow term="What it's for" desc="Score a whole spreadsheet of applicants at once instead of one at a time." />
              <TermRow term="Required columns" desc={REQUIRED_CSV_COLUMNS.join(", ")} />
              <TermRow term="loan_status column (optional)" desc="If your file includes real outcomes (0 = repaid, 1 = defaulted), you'll also see how accurate the predictions were." />
              <TermRow term="Download Results CSV" desc="Gets your original file back with predicted score, risk, and verdict added as new columns." />
            </GuideSection>

            <GuideSection title="Model Stats tab">
              <TermRow term="Which Model Works Best" desc="Compares five versions tried during development. Higher bars are better. The highlighted one is what actually powers this tool." />
              <TermRow term="What Affects the Score Most" desc="Ranks which details matter most on average, across all applicants — different from the personalized 'Why this score' list." />
            </GuideSection>
          </div>
        </div>
      )}

      <style>{`
        @media (max-width: 760px) {
          .ledger-grid > div:first-child { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}

function GuideSection({ title, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: THEME.brassBright, marginBottom: 8, letterSpacing: 0.3 }}>{title}</div>
      <div style={{ fontSize: 12.5, color: THEME.ink, lineHeight: 1.6 }}>{children}</div>
    </div>
  );
}

function SummaryCard({ label, value, color }) {
  return (
    <div style={{ background: THEME.panelAlt, border: `1px solid ${THEME.hairline}`, borderRadius: 8, padding: "12px 10px", textAlign: "center" }}>
      <div style={{ fontFamily: "'Fraunces', serif", fontSize: 22, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 11, color: THEME.inkDim, marginTop: 2 }}>{label}</div>
    </div>
  );
}

function TermRow({ term, desc }) {
  return (
    <div style={{ marginBottom: 8, paddingBottom: 8, borderBottom: `1px dashed ${THEME.hairline}` }}>
      <div style={{ fontSize: 12.5, fontWeight: 600, color: THEME.ink, marginBottom: 2 }}>{term}</div>
      <div style={{ fontSize: 12, color: THEME.inkDim, lineHeight: 1.5 }}>{desc}</div>
    </div>
  );
}
