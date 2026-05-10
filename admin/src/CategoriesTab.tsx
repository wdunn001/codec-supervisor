import type { Category, CategoryAction, InternalPolicy } from "./types";

const CATEGORY_NAME_RE = /^[a-z0-9_-]+$/;
const ACTIONS: CategoryAction[] = ["stop", "redact", "regenerate", "flag"];

/**
 * Categories tab — the per-category enforcement table.
 * `(name, action, description?)` rows; add / remove / reorder.
 *
 * Validation here is permissive (keeps the UI usable while the operator
 * is mid-edit). Final validation lives on the PUT endpoint, which uses
 * the same Pydantic regex to reject bad names. Save-on-error returns a
 * 400 / 422; the App-level toolbar surfaces it.
 */
export function CategoriesTab({
  draft,
  setDraft,
}: {
  draft: InternalPolicy;
  setDraft: (next: InternalPolicy) => void;
}): JSX.Element {
  const updateRow = (idx: number, patch: Partial<Category>) => {
    const next = draft.categories.slice();
    const existing = next[idx];
    if (!existing) return;
    next[idx] = { ...existing, ...patch };
    setDraft({ ...draft, categories: next });
  };

  const removeRow = (idx: number) => {
    if (draft.categories.length === 1) {
      // Schema requires at least one category — soft-block.
      return;
    }
    const next = draft.categories.slice();
    next.splice(idx, 1);
    setDraft({ ...draft, categories: next });
  };

  const addRow = () => {
    setDraft({
      ...draft,
      categories: [...draft.categories, { name: "", action: "stop" }],
    });
  };

  return (
    <div>
      <p className="muted">
        Each row is a category enforced by this policy. Names must match{" "}
        <code>[a-z0-9_-]+</code>. Actions:{" "}
        <code>stop</code> halts generation;{" "}
        <code>redact</code> replaces the offending span with policy-defined
        placeholder token IDs and continues;{" "}
        <code>regenerate</code> rewinds and resamples;{" "}
        <code>flag</code> annotates without intervening.
      </p>

      {draft.categories.map((cat, idx) => {
        const nameOk = CATEGORY_NAME_RE.test(cat.name);
        return (
          <div className="cat-row" key={`${idx}-${cat.name}`}>
            <input
              value={cat.name}
              placeholder="category_name"
              style={{
                borderColor: cat.name && !nameOk ? "var(--danger)" : undefined,
              }}
              onChange={(e) => updateRow(idx, { name: e.target.value })}
            />
            <select
              value={cat.action}
              onChange={(e) =>
                updateRow(idx, { action: e.target.value as CategoryAction })
              }
            >
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
            <input
              value={cat.description ?? ""}
              placeholder="optional description"
              onChange={(e) =>
                updateRow(idx, { description: e.target.value || null })
              }
            />
            <button
              className="delete"
              onClick={() => removeRow(idx)}
              disabled={draft.categories.length === 1}
              title={
                draft.categories.length === 1
                  ? "schema requires ≥1 category"
                  : "delete row"
              }
            >
              ×
            </button>
          </div>
        );
      })}

      <button onClick={addRow}>+ category</button>
    </div>
  );
}
