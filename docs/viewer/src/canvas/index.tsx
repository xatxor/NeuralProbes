import {
  type CSSProperties,
  type ReactNode,
  useState,
  type ElementType,
} from "react";
import { computeDAGLayout } from "./dag-layout";
import { darkTheme, useHostTheme } from "./theme";

export { computeDAGLayout, useHostTheme };

export function useCanvasState<T>(_key: string, defaultValue: T): [T, (v: T | ((p: T) => T)) => void] {
  return useState(defaultValue);
}
export type { DAGLayoutOptions, DAGLayoutResult } from "./dag-layout";

const toneColors = {
  success: "#3d9970",
  danger: "#c44",
  warning: "#c90",
  info: "#4da3ff",
  neutral: "#666",
};

type StyleProps = { style?: CSSProperties; children?: ReactNode };

export function Stack({ children, gap = 0, style }: StyleProps & { gap?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap, ...style }}>{children}</div>
  );
}

export function Row({
  children,
  gap = 0,
  align,
  wrap,
  style,
}: StyleProps & { gap?: number; align?: string; wrap?: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        alignItems: align,
        flexWrap: wrap ? "wrap" : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function H1({ children, style }: StyleProps) {
  return <h1 style={{ margin: 0, fontSize: 24, fontWeight: 600, ...style }}>{children}</h1>;
}

export function H2({ children, style }: StyleProps) {
  return <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, ...style }}>{children}</h2>;
}

export function H3({ children, style }: StyleProps) {
  return <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, ...style }}>{children}</h3>;
}

export function Text({
  children,
  style,
  as,
  weight,
}: StyleProps & { as?: ElementType; weight?: "normal" | "medium" | "semibold" | "bold" }) {
  const Tag = as ?? "p";
  const weights = { normal: 400, medium: 500, semibold: 600, bold: 700 };
  return (
    <Tag
      style={{
        margin: 0,
        fontWeight: weight ? weights[weight] : undefined,
        ...style,
      }}
    >
      {children}
    </Tag>
  );
}

export function Card({ children, style }: StyleProps) {
  return (
    <div
      style={{
        border: `1px solid ${darkTheme.stroke.primary}`,
        borderRadius: 6,
        background: darkTheme.bg.elevated,
        overflow: "hidden",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, trailing, style }: StyleProps & { trailing?: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 14px",
        borderBottom: `1px solid ${darkTheme.stroke.primary}`,
        fontSize: 13,
        fontWeight: 600,
        ...style,
      }}
    >
      <span>{children}</span>
      {trailing && (
        <span style={{ fontSize: 11, fontWeight: 400, color: darkTheme.text.tertiary }}>{trailing}</span>
      )}
    </div>
  );
}

export function CardBody({ children, style }: StyleProps) {
  return <div style={{ padding: 14, ...style }}>{children}</div>;
}

export function Select({
  value,
  onChange,
  options,
  style,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  style?: CSSProperties;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: "100%",
        padding: "8px 10px",
        borderRadius: 4,
        border: `1px solid ${darkTheme.stroke.primary}`,
        background: darkTheme.fill.tertiary,
        color: darkTheme.text.primary,
        fontSize: 13,
        ...style,
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Pill({
  children,
  size,
  style,
}: StyleProps & { size?: "sm" | "md" }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: size === "sm" ? "2px 8px" : "4px 10px",
        borderRadius: 4,
        fontSize: size === "sm" ? 11 : 12,
        border: `1px solid ${darkTheme.stroke.secondary}`,
        background: darkTheme.fill.tertiary,
        color: darkTheme.text.secondary,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  tone,
  style,
}: {
  label: string;
  value: string;
  tone?: keyof typeof toneColors;
  style?: CSSProperties;
}) {
  return (
    <div style={{ ...style }}>
      <div style={{ fontSize: 11, color: darkTheme.text.tertiary, marginBottom: 4 }}>{label}</div>
      <div
        style={{
          fontSize: 18,
          fontWeight: 600,
          color: tone ? toneColors[tone] : darkTheme.text.primary,
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function CollapsibleSection({
  title,
  trailing,
  children,
  defaultOpen = false,
  style,
}: StyleProps & { title: string; trailing?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ ...style }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 0",
          border: "none",
          background: "none",
          color: darkTheme.text.primary,
          cursor: "pointer",
          fontSize: 14,
          fontWeight: 600,
        }}
      >
        <span>{open ? "▾" : "▸"} {title}</span>
        {trailing && (
          <span style={{ fontSize: 11, fontWeight: 400, color: darkTheme.text.tertiary }}>{trailing}</span>
        )}
      </button>
      {open && <div>{children}</div>}
    </div>
  );
}

export function Callout({
  children,
  tone = "info",
  style,
}: StyleProps & { tone?: keyof typeof toneColors }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 4,
        border: `1px solid ${toneColors[tone]}55`,
        background: `${toneColors[tone]}18`,
        fontSize: 13,
        lineHeight: "20px",
        color: darkTheme.text.secondary,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Table({
  headers,
  rows,
  striped,
  style,
}: {
  headers: string[];
  rows: string[][];
  striped?: boolean;
  style?: CSSProperties;
}) {
  return (
    <div style={{ overflowX: "auto", ...style }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                style={{
                  textAlign: "left",
                  padding: "6px 8px",
                  borderBottom: `1px solid ${darkTheme.stroke.primary}`,
                  color: darkTheme.text.tertiary,
                  fontWeight: 500,
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              style={{
                background: striped && i % 2 ? darkTheme.fill.tertiary : undefined,
              }}
            >
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{
                    padding: "6px 8px",
                    borderBottom: `1px solid ${darkTheme.stroke.secondary}`,
                    color: darkTheme.text.secondary,
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type ChartSeries = {
  name: string;
  data: number[];
  tone?: keyof typeof toneColors;
};

export function BarChart({
  categories,
  series,
  height = 180,
  valueSuffix = "",
  showValues,
  referenceLines,
  style,
}: {
  categories: string[];
  series: ChartSeries[];
  height?: number;
  valueSuffix?: string;
  showValues?: boolean;
  referenceLines?: Array<{ value: number; label?: string }>;
  style?: CSSProperties;
}) {
  const s = series[0];
  if (!s) return null;
  const max = Math.max(...s.data, ...(referenceLines?.map((r) => r.value) ?? [0]), 0.01);
  const barW = Math.min(48, 320 / categories.length - 8);
  const chartW = categories.length * (barW + 12) + 40;
  const chartH = height - 40;

  return (
    <svg width="100%" viewBox={`0 0 ${Math.max(chartW, 280)} ${height}`} style={{ display: "block", ...style }}>
      {referenceLines?.map((line, i) => {
        const y = 20 + chartH - (line.value / max) * chartH;
        return (
          <g key={i}>
            <line
              x1={36}
              y1={y}
              x2={chartW}
              y2={y}
              stroke={darkTheme.text.tertiary}
              strokeDasharray="4 3"
            />
            {line.label && (
              <text x={4} y={y + 4} fill={darkTheme.text.tertiary} fontSize={9}>
                {line.label}
              </text>
            )}
          </g>
        );
      })}
      {categories.map((cat, i) => {
        const val = s.data[i] ?? 0;
        const h = (val / max) * chartH;
        const x = 40 + i * (barW + 12);
        const y = 20 + chartH - h;
        return (
          <g key={cat}>
            <rect x={x} y={y} width={barW} height={h} rx={2} fill={toneColors[s.tone ?? "info"]} opacity={0.85} />
            {showValues && (
              <text x={x + barW / 2} y={y - 4} textAnchor="middle" fill={darkTheme.text.secondary} fontSize={10}>
                {val}
                {valueSuffix}
              </text>
            )}
            <text
              x={x + barW / 2}
              y={height - 8}
              textAnchor="middle"
              fill={darkTheme.text.tertiary}
              fontSize={10}
            >
              {cat}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
