import { ReactNode, useMemo, useState } from "react";
import { downloadCsv } from "../lib/csv";

export interface Column<T> {
  key: string;
  label: string;
  render?: (row: T) => ReactNode;
  value?: (row: T) => unknown;   // used for sorting and CSV
  align?: "left" | "right";
  title?: string;
}

export function DataTable<T extends Record<string, unknown>>({ rows, columns, pageSize = 25, csvName, filter,
  emptyText = "No rows.", rowClass }: {
  rows: T[]; columns: Column<T>[]; pageSize?: number; csvName?: string; filter?: (row: T, q: string) => boolean;
  emptyText?: string; rowClass?: (row: T) => string;
}) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [asc, setAsc] = useState(true);
  const [page, setPage] = useState(0);
  const [q, setQ] = useState("");

  const val = (row: T, c: Column<T>) => (c.value ? c.value(row) : row[c.key]);
  const filtered = useMemo(() => (q && filter ? rows.filter((r) => filter(r, q.toLowerCase())) : rows), [rows, q, filter]);
  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    const col = columns.find((c) => c.key === sortKey);
    if (!col) return filtered;
    return [...filtered].sort((a, b) => {
      const x = val(a, col);
      const y = val(b, col);
      if (x === y) return 0;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x < y ? -1 : 1) * (asc ? 1 : -1);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtered, sortKey, asc, columns]);
  const pages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const p = Math.min(page, pages - 1);
  const view = sorted.slice(p * pageSize, (p + 1) * pageSize);

  const exportCsv = () => {
    const out = sorted.map((r) => Object.fromEntries(columns.map((c) => [c.key, val(r, c)])));
    downloadCsv(csvName ?? "table.csv", out, columns.map((c) => c.key));
  };

  return (
    <div className="table-wrap">
      <div className="table-tools">
        {filter && (
          <input type="search" placeholder="Filter rows…" value={q} aria-label="Filter rows"
            onChange={(e) => { setQ(e.target.value); setPage(0); }} />
        )}
        <span className="muted">{sorted.length.toLocaleString()} rows</span>
        {csvName && <button type="button" className="btn ghost" onClick={exportCsv}>Export CSV</button>}
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key} title={c.title} className={c.align === "right" ? "num" : ""}
                  aria-sort={sortKey === c.key ? (asc ? "ascending" : "descending") : "none"}>
                  <button type="button" onClick={() => {
                    if (sortKey === c.key) setAsc(!asc); else { setSortKey(c.key); setAsc(true); }
                  }}>
                    {c.label}{sortKey === c.key ? (asc ? " ▲" : " ▼") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.length === 0 && (
              <tr><td colSpan={columns.length} className="muted center">{emptyText}</td></tr>
            )}
            {view.map((r, i) => (
              <tr key={i} className={rowClass?.(r)}>
                {columns.map((c) => (
                  <td key={c.key} className={c.align === "right" ? "num" : ""}>
                    {c.render ? c.render(r) : String(val(r, c) ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="pager">
          <button type="button" className="btn ghost" disabled={p === 0} onClick={() => setPage(0)}>«</button>
          <button type="button" className="btn ghost" disabled={p === 0} onClick={() => setPage(p - 1)}>‹</button>
          <span className="muted">Page {p + 1} of {pages}</span>
          <button type="button" className="btn ghost" disabled={p >= pages - 1} onClick={() => setPage(p + 1)}>›</button>
          <button type="button" className="btn ghost" disabled={p >= pages - 1} onClick={() => setPage(pages - 1)}>»</button>
        </div>
      )}
    </div>
  );
}
