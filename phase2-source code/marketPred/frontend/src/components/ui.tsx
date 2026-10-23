import { ReactNode } from "react";

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }: {
  label: string; value: ReactNode; hint?: ReactNode; tone?: "good" | "bad" | "warn" | "muted";
}) {
  return (
    <div className="stat" title={typeof hint === "string" ? hint : undefined}>
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ""}`}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral", title }: {
  children: ReactNode; tone?: "neutral" | "actual" | "forecast" | "good" | "bad" | "warn"; title?: string;
}) {
  return <span className={`badge ${tone}`} title={title}>{children}</span>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state loading" role="status">
      <span className="spinner" aria-hidden /> {label}
    </div>
  );
}

export function ErrorBox({ error, hint }: { error: string; hint?: ReactNode }) {
  return (
    <div className="state error" role="alert">
      <strong>Unavailable.</strong> {error}
      {hint && <div className="state-hint">{hint}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="state empty">{children}</div>;
}

export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return <div className={`note ${tone}`}>{children}</div>;
}

export function Field({ label, children, help }: { label: string; children: ReactNode; help?: string }) {
  return (
    <label className="field" title={help}>
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

export function Select<T extends string>({ value, onChange, options, disabled, ariaLabel }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string; disabled?: boolean }[];
  disabled?: boolean; ariaLabel?: string;
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as T)} disabled={disabled} aria-label={ariaLabel}>
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
      ))}
    </select>
  );
}

export function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string; disabled?: boolean; title?: string }[];
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button key={o.value} type="button" role="tab" aria-selected={value === o.value}
          className={value === o.value ? "on" : ""} disabled={o.disabled} title={o.title}
          onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Legend chip used to make actual vs forecast styling explicit. */
export function SeriesKey({ kind, label }: { kind: "actual" | "forecast" | "baseline"; label: string }) {
  return <span className={`series-key ${kind}`}><i aria-hidden />{label}</span>;
}
