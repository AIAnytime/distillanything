import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useToast } from "../components";
import type { DatasetInfo } from "../types";

// Renders the recipe the server will write — same shape as `distill init` YAML,
// so what you see here is exactly what lands in runs/<name>/recipe.yaml.
function toYaml(value: unknown, indent = 0): string {
  const pad = "  ".repeat(indent);
  if (value === null || value === undefined) return "null";
  if (typeof value !== "object") return String(value);
  return Object.entries(value as Record<string, unknown>)
    .map(([k, v]) =>
      v !== null && typeof v === "object"
        ? `${pad}${k}:\n${toYaml(v, indent + 1)}`
        : `${pad}${k}: ${toYaml(v)}`,
    )
    .join("\n");
}

export function NewRunPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [busy, setBusy] = useState(false);

  const [name, setName] = useState("");
  const [dataset, setDataset] = useState("");
  const [mode, setMode] = useState<"logit" | "seqkd">("logit");
  const [teacher, setTeacher] = useState("hf:HuggingFaceTB/SmolLM2-360M-Instruct");
  const [student, setStudent] = useState("HuggingFaceTB/SmolLM2-135M-Instruct");
  const [useLora, setUseLora] = useState(false);
  const [loraR, setLoraR] = useState(16);
  const [lossKind, setLossKind] = useState("forward_kl");
  const [temperature, setTemperature] = useState(2.0);
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState<number | "">(256);
  const [lr, setLr] = useState(1e-4);
  const [epochs, setEpochs] = useState(2);
  const [maxSteps, setMaxSteps] = useState<number | "">("");
  const [batchSize, setBatchSize] = useState(2);
  const [gradAccum, setGradAccum] = useState(8);
  const [gradCkpt, setGradCkpt] = useState(false);

  useEffect(() => {
    api<DatasetInfo[]>("/api/datasets").then((ds) => {
      setDatasets(ds);
      if (ds.length && !dataset) setDataset(ds[0].name);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const config = useMemo(() => {
    const cfg: Record<string, unknown> = {
      mode,
      teacher: { spec: teacher },
      student: {
        model: student,
        ...(useLora ? { lora: { r: loraR, alpha: loraR * 2 } } : {}),
      },
      loss: {
        kind: lossKind,
        temperature,
        alpha,
        ...(topK !== "" ? { top_k: topK } : {}),
      },
      train: {
        lr,
        epochs,
        ...(maxSteps !== "" ? { max_steps: maxSteps } : {}),
        batch_size: batchSize,
        grad_accum: gradAccum,
        gradient_checkpointing: gradCkpt,
      },
    };
    return cfg;
  }, [mode, teacher, student, useLora, loraR, lossKind, temperature, alpha, topK, lr, epochs, maxSteps, batchSize, gradAccum, gradCkpt]);

  const logitMismatch = mode === "logit" && !teacher.startsWith("hf:");

  const submit = async () => {
    setBusy(true);
    try {
      await api("/api/jobs/train", {
        method: "POST",
        body: JSON.stringify({ name, dataset, config }),
      });
      toast(`Run ${name} submitted`);
      navigate(`/runs/${name}`);
    } catch (e) {
      toast(`Submit failed: ${(e as Error).message}`, true);
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>New run</h1>
          <div className="sub">
            Everything here is a recipe — the exact YAML is saved into the run directory for
            reproducibility.
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16 }}>
        <div style={{ display: "grid", gap: 16, alignContent: "start" }}>
          <div className="card">
            <div className="card-head">Run</div>
            <div className="card-body grid-2">
              <div className="field">
                <label>Run name</label>
                <input
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="mac-small-v2"
                />
                <span className="hint">output lands in runs/&lt;name&gt;</span>
              </div>
              <div className="field">
                <label>Dataset</label>
                <select className="select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  {datasets.map((ds) => (
                    <option key={ds.name} value={ds.name}>
                      {ds.name} ({ds.records} records)
                    </option>
                  ))}
                </select>
                <span className="hint">
                  none here? Generate one on the Datasets page first
                </span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head">Teacher → Student</div>
            <div className="card-body" style={{ display: "grid", gap: 14 }}>
              <div className="grid-2">
                <div className="field">
                  <label>Teacher spec</label>
                  <input className="input mono" value={teacher} onChange={(e) => setTeacher(e.target.value)} />
                  <span className="hint">hf:&lt;repo&gt; · claude · openai:&lt;m&gt; · ollama:&lt;m&gt;</span>
                </div>
                <div className="field">
                  <label>Student model</label>
                  <input className="input mono" value={student} onChange={(e) => setStudent(e.target.value)} />
                </div>
              </div>
              <div className="grid-2">
                <div className="field">
                  <label>KD mode</label>
                  <select
                    className="select"
                    value={mode}
                    onChange={(e) => setMode(e.target.value as "logit" | "seqkd")}
                  >
                    <option value="logit">logit — white-box (local teacher, shared tokenizer)</option>
                    <option value="seqkd">seqkd — black-box (any teacher, incl. APIs)</option>
                  </select>
                  {logitMismatch && (
                    <span className="hint" style={{ color: "var(--amber)" }}>
                      logit KD needs a local hf: teacher — switch to seqkd for API teachers
                    </span>
                  )}
                </div>
                <div className="field">
                  <label className="checks" style={{ marginTop: 22 }}>
                    <input type="checkbox" checked={useLora} onChange={(e) => setUseLora(e.target.checked)} />
                    Train LoRA adapters (1-3B students on 16GB)
                  </label>
                  {useLora && (
                    <div className="grid-2" style={{ marginTop: 6 }}>
                      <div className="field">
                        <label>rank r</label>
                        <input
                          className="input"
                          type="number"
                          min={1}
                          value={loraR}
                          onChange={(e) => setLoraR(Number(e.target.value))}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head">Loss</div>
            <div className="card-body grid-4">
              <div className="field">
                <label>Kind</label>
                <select className="select" value={lossKind} onChange={(e) => setLossKind(e.target.value)}>
                  <option value="forward_kl">forward_kl</option>
                  <option value="reverse_kl">reverse_kl</option>
                  <option value="jsd">jsd</option>
                </select>
              </div>
              <div className="field">
                <label>Temperature</label>
                <input className="input" type="number" step={0.5} value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Alpha (KD ↔ CE)</label>
                <input className="input" type="number" step={0.1} min={0} max={1} value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Top-k logits</label>
                <input className="input" type="number" value={topK}
                  onChange={(e) => setTopK(e.target.value === "" ? "" : Number(e.target.value))} />
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head">Training</div>
            <div className="card-body grid-4">
              <div className="field">
                <label>Learning rate</label>
                <input className="input" type="number" step="any" value={lr}
                  onChange={(e) => setLr(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Epochs</label>
                <input className="input" type="number" min={1} value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Max steps (optional)</label>
                <input className="input" type="number" value={maxSteps}
                  onChange={(e) => setMaxSteps(e.target.value === "" ? "" : Number(e.target.value))} />
              </div>
              <div className="field">
                <label>Batch × accum</label>
                <div style={{ display: "flex", gap: 6 }}>
                  <input className="input" type="number" min={1} value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))} />
                  <input className="input" type="number" min={1} value={gradAccum}
                    onChange={(e) => setGradAccum(Number(e.target.value))} />
                </div>
              </div>
              <label className="checks" style={{ gridColumn: "span 2" }}>
                <input type="checkbox" checked={gradCkpt} onChange={(e) => setGradCkpt(e.target.checked)} />
                Gradient checkpointing (recommended for LoRA ≥1B on laptops)
              </label>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gap: 16, alignContent: "start" }}>
          <div className="card">
            <div className="card-head">recipe.yaml preview</div>
            <pre className="logpane" style={{ margin: 0, maxHeight: 420 }}>
              {toYaml(config)}
            </pre>
          </div>
          <button
            className="btn btn-primary"
            style={{ justifyContent: "center", padding: "10px 0", fontSize: 14 }}
            disabled={busy || !name || !dataset}
            onClick={submit}
          >
            ▶ Start training
          </button>
          <div className="hint" style={{ color: "var(--muted)", fontSize: 12 }}>
            One training job runs at a time (16GB-laptop default) — extra submissions queue up.
            The run appears on the Runs page immediately and streams live.
          </div>
        </div>
      </div>
    </div>
  );
}
