export interface MetricPoint {
  kind: "train" | "eval";
  step: number;
  loss?: number;
  kd?: number;
  ce?: number;
  lr?: number;
  elapsed_s?: number;
  eval_loss?: number;
  perplexity?: number;
  teacher_agreement?: number;
}

export interface RunSummary {
  name: string;
  state: string;
  mode: string | null;
  student: string | null;
  teacher: string | null;
  total_steps: number | null;
  steps_completed: number | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  last_metric: MetricPoint | null;
  has_report: boolean;
  quality_retention: number | null;
  updated_at: number | null;
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown> | null;
  eval: Record<string, number> | null;
}

export interface JudgeSummary {
  n: number;
  student_wins: number;
  ties: number;
  teacher_wins: number;
  student_win_rate: number;
  tie_rate: number;
  teacher_win_rate: number;
  quality_retention: number;
}

export interface Benchmark {
  parameters_m?: number;
  tokens_per_s?: number;
  latency_p50_s?: number;
  latency_p95_s?: number;
  memory_mb?: number;
  disk_size_mb?: number;
  cost_per_1k_tokens_usd?: number;
}

export interface Report {
  student_name: string;
  teacher_name: string | null;
  dataset: string;
  generated_on: string;
  n_prompts: number;
  judge?: JudgeSummary;
  judge_name?: string;
  train_eval?: Record<string, number>;
  student_benchmark?: Benchmark;
  teacher_benchmark?: Benchmark;
  samples?: { prompt: string; student_answer: string }[];
}

export interface DatasetInfo {
  name: string;
  records: number;
  size_bytes: number;
  updated_at: number;
}

export interface DatasetPage {
  name: string;
  total: number;
  offset: number;
  records: Record<string, unknown>[];
}

export interface Job {
  id: string;
  kind: string;
  run_name: string | null;
  status: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  returncode: number | null;
  pid: number | null;
}
