import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { MetricPoint } from "./types";

export interface SeriesSpec {
  key: keyof MetricPoint;
  label: string;
  color: string;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Streaming line chart over train metrics; steps on x. One uPlot instance,
 * setData on updates — cheap enough to redraw every SSE tick. */
export function MetricChart({
  points,
  series,
  height = 220,
  logY = false,
  runsData,
}: {
  points?: MetricPoint[];
  series: SeriesSpec[];
  height?: number;
  logY?: boolean;
  /** compare mode: one line per run, series[i] pairs with runsData[i] */
  runsData?: { points: MetricPoint[] }[];
}) {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  const buildData = (): uPlot.AlignedData => {
    if (runsData) {
      const stepSet = new Set<number>();
      runsData.forEach((r) =>
        r.points.forEach((p) => p.kind === "train" && stepSet.add(p.step)),
      );
      const steps = [...stepSet].sort((a, b) => a - b);
      const index = new Map(steps.map((s, i) => [s, i]));
      const cols = runsData.map((r) => {
        const col: (number | null)[] = steps.map(() => null);
        r.points.forEach((p) => {
          if (p.kind === "train" && p.loss !== undefined) col[index.get(p.step)!] = p.loss;
        });
        return col;
      });
      return [steps, ...cols] as uPlot.AlignedData;
    }
    const train = (points ?? []).filter((p) => p.kind === "train");
    const steps = train.map((p) => p.step);
    const cols = series.map((s) => train.map((p) => (p[s.key] as number | undefined) ?? null));
    return [steps, ...cols] as uPlot.AlignedData;
  };

  useEffect(() => {
    if (!el.current) return;
    const muted = cssVar("--muted");
    const grid = cssVar("--border");
    const opts: uPlot.Options = {
      width: el.current.clientWidth || 600,
      height,
      scales: {
        x: { time: false },
        y: logY ? { distr: 3 } : {},
      },
      axes: [
        {
          stroke: muted,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          font: "11px ui-monospace, Menlo, monospace",
        },
        {
          stroke: muted,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          font: "11px ui-monospace, Menlo, monospace",
          size: 56,
        },
      ],
      series: [
        { label: "step" },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: 1.6,
          points: { show: false },
        })),
      ],
      cursor: { drag: { x: true, y: false } },
      legend: { live: true },
    };
    plot.current = new uPlot(opts, buildData(), el.current);

    const ro = new ResizeObserver(() => {
      if (el.current && plot.current)
        plot.current.setSize({ width: el.current.clientWidth, height });
    });
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.map((s) => s.label).join(","), height, logY, runsData?.length]);

  useEffect(() => {
    plot.current?.setData(buildData());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, runsData]);

  return <div ref={el} />;
}

export const chartColors = {
  loss: "#2dd4bf",
  kd: "#f0b429",
  ce: "#58a6ff",
  lr: "#bc8cff",
  palette: ["#2dd4bf", "#f0b429", "#58a6ff", "#bc8cff", "#f85149", "#3fb950", "#ff9bce"],
};
