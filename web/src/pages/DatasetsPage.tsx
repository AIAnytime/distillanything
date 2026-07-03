import { useEffect, useState } from "react";
import { api } from "../api";
import { EmptyState, fmtBytes, timeAgo, usePoll, useToast } from "../components";
import type { DatasetInfo, DatasetPage } from "../types";

const PAGE_SIZE = 25;

function GenerateForm({ onSubmitted }: { onSubmitted: () => void }) {
  const [name, setName] = useState("");
  const [teacher, setTeacher] = useState("hf:HuggingFaceTB/SmolLM2-360M-Instruct");
  const [seeds, setSeeds] = useState("");
  const [expand, setExpand] = useState(0);
  const [judge, setJudge] = useState("");
  const [minScore, setMinScore] = useState(7);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const submit = async () => {
    setBusy(true);
    try {
      await api("/api/jobs/generate", {
        method: "POST",
        body: JSON.stringify({
          name,
          teacher,
          seeds_text: seeds,
          expand,
          ...(judge ? { judge, min_score: minScore } : {}),
        }),
      });
      toast(`Generating ${name}.jsonl with ${teacher} — watch it in Jobs.`);
      onSubmitted();
    } catch (e) {
      toast(`Generate failed: ${(e as Error).message}`, true);
    } finally {
      setBusy(false);
    }
  };

  const seedCount = seeds.split("\n").filter((s) => s.trim()).length;

  return (
    <div className="card-body" style={{ display: "grid", gap: 14 }}>
      <div className="grid-3">
        <div className="field">
          <label>Dataset name</label>
          <input
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-task"
          />
          <span className="hint">saved as data/&lt;name&gt;.jsonl</span>
        </div>
        <div className="field">
          <label>Teacher</label>
          <input className="input mono" value={teacher} onChange={(e) => setTeacher(e.target.value)} />
          <span className="hint">hf:&lt;repo&gt; · claude · openai:&lt;m&gt; · ollama:&lt;m&gt;</span>
        </div>
        <div className="field">
          <label>Expand per seed (self-instruct)</label>
          <input
            className="input"
            type="number"
            min={0}
            max={50}
            value={expand}
            onChange={(e) => setExpand(Number(e.target.value))}
          />
        </div>
      </div>
      <div className="field">
        <label>Seed prompts — one per line ({seedCount})</label>
        <textarea
          className="textarea"
          rows={6}
          value={seeds}
          onChange={(e) => setSeeds(e.target.value)}
          placeholder={"Explain knowledge distillation in two sentences.\nWrite a haiku about small models."}
        />
      </div>
      <div className="grid-3" style={{ alignItems: "end" }}>
        <div className="field">
          <label>Judge (optional quality gate)</label>
          <input
            className="input mono"
            value={judge}
            onChange={(e) => setJudge(e.target.value)}
            placeholder="claude"
          />
        </div>
        <div className="field">
          <label>Min score (1–10)</label>
          <input
            className="input"
            type="number"
            min={1}
            max={10}
            value={minScore}
            disabled={!judge}
            onChange={(e) => setMinScore(Number(e.target.value))}
          />
        </div>
        <div>
          <button
            className="btn btn-primary"
            disabled={busy || !name || !teacher || seedCount === 0}
            onClick={submit}
          >
            Generate dataset
          </button>
        </div>
      </div>
      <div className="hint" style={{ color: "var(--muted)", fontSize: 12 }}>
        API teachers read their keys from the environment where <code>distill ui</code> runs — the
        dashboard never asks for or stores keys.
      </div>
    </div>
  );
}

export function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[] | null>(null);
  const [showGenerate, setShowGenerate] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [page, setPage] = useState<DatasetPage | null>(null);
  const [offset, setOffset] = useState(0);

  usePoll(async () => {
    setDatasets(await api<DatasetInfo[]>("/api/datasets"));
  }, 5000);

  useEffect(() => {
    if (active) {
      api<DatasetPage>(`/api/datasets/${active}/records?offset=${offset}&limit=${PAGE_SIZE}`).then(
        setPage,
      );
    }
  }, [active, offset]);

  const open = (name: string) => {
    setActive(name);
    setOffset(0);
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Datasets</h1>
          <div className="sub">Curated JSONL under your data directory — readable, editable, versionable.</div>
        </div>
        <button className="btn btn-primary" onClick={() => setShowGenerate((v) => !v)}>
          {showGenerate ? "Close" : "Generate new"}
        </button>
      </div>

      {showGenerate && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-head">Generate a dataset with a teacher</div>
          <GenerateForm onSubmitted={() => setShowGenerate(false)} />
        </div>
      )}

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Dataset</th>
              <th className="right">Records</th>
              <th className="right">Size</th>
              <th className="right">Updated</th>
            </tr>
          </thead>
          <tbody>
            {datasets?.map((ds) => (
              <tr key={ds.name} className="rowlink" onClick={() => open(ds.name)}>
                <td>
                  <strong>{ds.name}</strong>
                  {active === ds.name && <span className="badge live" style={{ marginLeft: 8 }}>viewing</span>}
                </td>
                <td className="right num">{ds.records}</td>
                <td className="right num">{fmtBytes(ds.size_bytes)}</td>
                <td className="right" style={{ color: "var(--muted)" }}>
                  {timeAgo(ds.updated_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {datasets !== null && datasets.length === 0 && (
          <EmptyState
            title="No datasets yet"
            command="distill generate seeds.txt --teacher claude --out data/train.jsonl"
          >
            Generate one with a teacher — here or from the CLI.
          </EmptyState>
        )}
      </div>

      {active && page && (
        <div className="card">
          <div className="card-head">
            {active}
            <span style={{ color: "var(--muted)", fontWeight: 400 }}>
              {page.offset + 1}–{Math.min(page.offset + PAGE_SIZE, page.total)} of {page.total}
              <button
                className="btn btn-sm"
                style={{ marginLeft: 12 }}
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                ‹ Prev
              </button>
              <button
                className="btn btn-sm"
                style={{ marginLeft: 6 }}
                disabled={offset + PAGE_SIZE >= page.total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next ›
              </button>
            </span>
          </div>
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: "40%" }}>Prompt</th>
                <th>Response</th>
                <th className="right">Score</th>
              </tr>
            </thead>
            <tbody>
              {page.records.map((rec, i) => (
                <tr key={i}>
                  <td style={{ whiteSpace: "pre-wrap" }}>{String(rec.prompt ?? rec.text ?? "—")}</td>
                  <td style={{ whiteSpace: "pre-wrap", color: "var(--muted)" }}>
                    {String(rec.response ?? "—").slice(0, 400)}
                  </td>
                  <td className="right num">{rec.judge_score !== undefined ? String(rec.judge_score) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
