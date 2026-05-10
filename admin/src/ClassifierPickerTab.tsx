import { useEffect, useState } from "react";
import { ApiError, listClassifiers } from "./api";
import type { ClassifierEntry, InternalPolicy } from "./types";

/**
 * ClassifierPicker tab — shows the classifiers registered with the
 * supervisor (`GET /admin/policies/_classifiers`) and lets the operator
 * bind one to this policy by writing its `model_id` into the policy's
 * `classifier.family` field.
 *
 * Server-side registry is the source of truth; the admin UI doesn't
 * hardcode classifier names. Hosts that ship their own classifiers
 * (custom `register(...)` calls) automatically appear here.
 */
export function ClassifierPickerTab({
  draft,
  setDraft,
}: {
  draft: InternalPolicy;
  setDraft: (next: InternalPolicy) => void;
}): JSX.Element {
  const [entries, setEntries] = useState<ClassifierEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listClassifiers()
      .then((rows) => {
        if (!cancelled) setEntries(rows);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof ApiError ? `${e.status}: ${e.message}` : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = draft.classifier.family;

  const select = (modelId: string) => {
    setDraft({
      ...draft,
      classifier: { ...draft.classifier, family: modelId },
    });
  };

  return (
    <div>
      <p className="muted">
        Pick the classifier the supervisor will run for semantic
        enforcement (Layer 3). Bound to{" "}
        <code>classifier.family</code> on this policy. The descriptor
        published at <code>.well-known/codec/policies/&lt;id&gt;.json</code>{" "}
        discloses the family but never the thresholds.
      </p>

      {error ? <div className="error">{error}</div> : null}

      {entries === null ? (
        <p className="muted">loading classifiers…</p>
      ) : entries.length === 0 ? (
        <p className="muted">
          No classifiers registered. Either{" "}
          <code>safety_classifiers.register_default_classifiers()</code> hasn't
          run, or the supervisor was started without the{" "}
          <code>classifiers</code> extra installed.
        </p>
      ) : (
        <div>
          {entries.map((c) => {
            const active = c.model_id === selected;
            return (
              <div
                key={c.model_id}
                className="classifier-card"
                style={{
                  border: "1px solid var(--border)",
                  background: active ? "var(--panel)" : "transparent",
                  borderColor: active ? "var(--accent)" : "var(--border)",
                  padding: "10px 12px",
                  marginBottom: 8,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <div>
                    <strong style={{ color: active ? "var(--accent)" : "var(--fg)" }}>
                      {c.model_id}
                    </strong>{" "}
                    <span className="muted">tier {c.tier} · requires {c.requires}</span>
                  </div>
                  {active ? (
                    <span style={{ color: "var(--ok)" }}>● selected</span>
                  ) : (
                    <button onClick={() => select(c.model_id)}>select</button>
                  )}
                </div>
                {c.description ? (
                  <div style={{ marginTop: 4 }}>{c.description}</div>
                ) : null}
                <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                  categories: {c.categories.join(", ") || "(none advertised)"}
                </div>
              </div>
            );
          })}

          <p className="muted" style={{ marginTop: 12 }}>
            Current binding: <code>{selected || "(unset)"}</code>
          </p>
        </div>
      )}
    </div>
  );
}
