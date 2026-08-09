interface Metric {
  label: string;
  value?: number | null;
  /** 低于该值视为告警 */
  warnBelow?: number;
}

function format(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export function MetricBadges({ metrics }: { metrics: Metric[] }) {
  const shown = metrics.filter(
    (m) => m.value !== null && m.value !== undefined,
  );
  if (!shown.length) return null;

  return (
    <div className="metric-badges">
      {shown.map((m) => {
        const warn =
          m.warnBelow !== undefined &&
          typeof m.value === "number" &&
          m.value < m.warnBelow;
        return (
          <span
            key={m.label}
            className={`badge ${warn ? "badge-warn" : "badge-ok"}`}
          >
            {m.label}: {format(m.value)}
          </span>
        );
      })}
    </div>
  );
}
