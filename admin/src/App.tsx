import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  deletePolicy,
  getPolicy,
  listPolicies,
  putPolicy,
  sanitizePolicy,
} from "./api";
import { CategoriesTab } from "./CategoriesTab";
import { EditorTab } from "./EditorTab";
import type { InternalPolicy, SanitizeResponse } from "./types";

/**
 * Admin shell. Slice 5 ships TWO tabs (Editor + Categories) for each
 * selected policy; later slices add Classifier picker, Test bench,
 * Versions, and the .well-known generator.
 */
type Tab = "editor" | "categories";

const EMPTY_POLICY: InternalPolicy = {
  id: "",
  version: "1",
  tokenizers: ["meta-llama/llama-3"],
  categories: [{ name: "secrets", action: "stop" }],
  classifier: { family: "llama-guard-3-1b", host: "server" },
};

export function App(): JSX.Element {
  const [ids, setIds] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<InternalPolicy | null>(null);
  const [tab, setTab] = useState<Tab>("editor");
  const [lastSanitize, setLastSanitize] = useState<SanitizeResponse | null>(null);
  const [saving, setSaving] = useState(false);

  // Fetch the policy list on mount + after every mutation.
  const refreshList = useCallback(async () => {
    try {
      setIds(await listPolicies());
    } catch (e) {
      setError(formatError(e));
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  const selectId = useCallback(
    async (id: string | null) => {
      setError(null);
      setSelectedId(id);
      setLastSanitize(null);
      if (id === null) {
        setDraft(null);
        return;
      }
      try {
        setDraft(await getPolicy(id));
      } catch (e) {
        setError(formatError(e));
        setDraft(null);
      }
    },
    [],
  );

  const startNew = useCallback(() => {
    setError(null);
    setSelectedId(null);
    setLastSanitize(null);
    setDraft({ ...EMPTY_POLICY });
    setTab("editor");
  }, []);

  const save = useCallback(async () => {
    if (!draft) return;
    setError(null);
    setSaving(true);
    try {
      const result = await putPolicy(draft.id, draft);
      setLastSanitize(result);
      setSelectedId(draft.id);
      await refreshList();
    } catch (e) {
      setError(formatError(e));
    } finally {
      setSaving(false);
    }
  }, [draft, refreshList]);

  const sanitize = useCallback(async () => {
    if (!selectedId) return;
    setError(null);
    try {
      setLastSanitize(await sanitizePolicy(selectedId));
    } catch (e) {
      setError(formatError(e));
    }
  }, [selectedId]);

  const remove = useCallback(async () => {
    if (!selectedId) return;
    if (!window.confirm(`Delete policy "${selectedId}"? Version history is kept.`)) return;
    setError(null);
    try {
      await deletePolicy(selectedId);
      setSelectedId(null);
      setDraft(null);
      setLastSanitize(null);
      await refreshList();
    } catch (e) {
      setError(formatError(e));
    }
  }, [refreshList, selectedId]);

  const summary = useMemo(() => {
    if (!lastSanitize) return null;
    const rs = lastSanitize.descriptor.rules_summary ?? {};
    return (
      <div className="summary">
        <div>
          last sanitize → <code>{lastSanitize.hash}</code>
        </div>
        <div className="muted">
          banned-ids: {rs.banned_token_id_count ?? 0} · regex: {rs.regex_pattern_count ?? 0}{" "}
          · grammar: {rs.grammar_constraint_count ?? 0} · multi-token:{" "}
          {rs.multi_token_pattern_count ?? 0}
        </div>
      </div>
    );
  }, [lastSanitize]);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Safety policies</h1>
        {ids === null ? (
          <p className="muted">loading…</p>
        ) : ids.length === 0 ? (
          <p className="muted">none yet</p>
        ) : (
          <ul>
            {ids.map((id) => (
              <li key={id}>
                <button
                  className={id === selectedId ? "active" : ""}
                  onClick={() => void selectId(id)}
                >
                  {id}
                </button>
              </li>
            ))}
          </ul>
        )}
        <button className="new-policy" onClick={startNew}>
          + new policy
        </button>
      </aside>

      <main className="workspace">
        {error ? <div className="error">{error}</div> : null}

        {!draft ? (
          <p className="muted">
            Select a policy on the left, or click <strong>+ new policy</strong> to start one.
          </p>
        ) : (
          <>
            <div className="tabs">
              <button
                className={tab === "editor" ? "active" : ""}
                onClick={() => setTab("editor")}
              >
                Editor
              </button>
              <button
                className={tab === "categories" ? "active" : ""}
                onClick={() => setTab("categories")}
              >
                Categories
              </button>
            </div>

            <div className="toolbar">
              <button className="save" onClick={() => void save()} disabled={saving}>
                {saving ? "saving…" : "save"}
              </button>
              <button onClick={() => void sanitize()} disabled={!selectedId}>
                sanitize
              </button>
              {selectedId ? (
                <button className="danger" onClick={() => void remove()}>
                  delete
                </button>
              ) : null}
            </div>

            {tab === "editor" ? (
              <EditorTab draft={draft} setDraft={setDraft} />
            ) : (
              <CategoriesTab draft={draft} setDraft={setDraft} />
            )}

            {summary}
          </>
        )}
      </main>
    </div>
  );
}

function formatError(e: unknown): string {
  if (e instanceof ApiError) return `${e.status}: ${e.message}`;
  if (e instanceof Error) return e.message;
  return String(e);
}
