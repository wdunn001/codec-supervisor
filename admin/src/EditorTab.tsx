import type { ChangeEvent } from "react";
import type { InternalPolicy } from "./types";

/**
 * Editor tab — non-categories metadata. ID, version, tokenizer bindings,
 * classifier family/host/feature requirements, optional category-registry
 * pointer + publisher block. Categories are separate (CategoriesTab).
 *
 * Internal-only fields (banned_token_ids etc.) live raw-JSON in this tab
 * for slice 5 — slices 6+ replace the textareas with proper editors
 * (banned-id picker, regex tester, etc.). Operators today edit them as
 * arrays in the textarea; the PUT validates shape on save.
 */
export function EditorTab({
  draft,
  setDraft,
}: {
  draft: InternalPolicy;
  setDraft: (next: InternalPolicy) => void;
}): JSX.Element {
  const update = <K extends keyof InternalPolicy>(key: K, value: InternalPolicy[K]) => {
    setDraft({ ...draft, [key]: value });
  };

  const onText = (key: keyof InternalPolicy) => (e: ChangeEvent<HTMLInputElement>) => {
    update(key, e.target.value as InternalPolicy[typeof key]);
  };

  const tokenizersText = draft.tokenizers.join(", ");
  const onTokenizersChange = (e: ChangeEvent<HTMLInputElement>) => {
    update(
      "tokenizers",
      e.target.value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    );
  };

  const features = draft.classifier.requires_engine_features ?? [];

  return (
    <div>
      <section>
        <h3>Identity</h3>
        <div className="field-row">
          <label>id</label>
          <input
            value={draft.id}
            onChange={onText("id")}
            placeholder="acme/strict-v3"
          />
        </div>
        <div className="field-row">
          <label>version</label>
          <input value={draft.version} onChange={onText("version")} />
        </div>
        <div className="field-row">
          <label>tokenizers</label>
          <input
            value={tokenizersText}
            onChange={onTokenizersChange}
            placeholder="meta-llama/llama-3, qwen/qwen2"
          />
        </div>
        <div className="field-row">
          <label>category registry</label>
          <input
            value={draft.category_registry ?? ""}
            onChange={(e) =>
              update("category_registry", e.target.value || null)
            }
            placeholder="mlcommons/ailuminate-v0.5 (optional)"
          />
        </div>
      </section>

      <section>
        <h3>Classifier</h3>
        <div className="field-row">
          <label>family</label>
          <input
            value={draft.classifier.family}
            onChange={(e) =>
              update("classifier", { ...draft.classifier, family: e.target.value })
            }
          />
        </div>
        <div className="field-row">
          <label>host</label>
          <select
            value={draft.classifier.host ?? "server"}
            onChange={(e) =>
              update("classifier", {
                ...draft.classifier,
                host: e.target.value as "server" | "client" | "both",
              })
            }
          >
            <option value="server">server</option>
            <option value="client">client</option>
            <option value="both">both</option>
          </select>
        </div>
        <div className="field-row">
          <label>engine features</label>
          <div>
            {(["logits_processor", "hidden_states", "sampling_chain"] as const).map(
              (feat) => (
                <label key={feat} style={{ marginRight: 12 }}>
                  <input
                    type="checkbox"
                    style={{ width: "auto", marginRight: 4 }}
                    checked={features.includes(feat)}
                    onChange={(e) => {
                      const next = new Set(features);
                      if (e.target.checked) next.add(feat);
                      else next.delete(feat);
                      update("classifier", {
                        ...draft.classifier,
                        requires_engine_features:
                          next.size === 0 ? null : Array.from(next),
                      });
                    }}
                  />
                  {feat}
                </label>
              ),
            )}
          </div>
        </div>
      </section>

      <section>
        <h3>Internal-only payloads</h3>
        <p className="muted">
          These fields are stripped by <code>sanitize</code>. Slice 5 ships raw
          textareas; slices 6+ replace them with category-specific editors.
        </p>
        <RawJsonField
          label="banned_token_ids"
          value={draft.banned_token_ids ?? null}
          placeholder="[4218, 5544, 9012]"
          onChange={(parsed) =>
            update("banned_token_ids", parsed as number[] | null)
          }
        />
        <RawJsonField
          label="regex_patterns"
          value={draft.regex_patterns ?? null}
          placeholder='["AKIA[0-9A-Z]{16}", "ghp_[A-Za-z0-9]{36}"]'
          onChange={(parsed) =>
            update("regex_patterns", parsed as string[] | null)
          }
        />
        <RawJsonField
          label="multi_token_patterns"
          value={draft.multi_token_patterns ?? null}
          placeholder='[{"literal": "secret-string-1"}]'
          onChange={(parsed) =>
            update(
              "multi_token_patterns",
              parsed as Array<Record<string, unknown>> | null,
            )
          }
        />
      </section>

      <section>
        <h3>Publisher (optional)</h3>
        <div className="field-row">
          <label>name</label>
          <input
            value={draft.publisher?.name ?? ""}
            onChange={(e) =>
              update("publisher", {
                ...(draft.publisher ?? {}),
                name: e.target.value || null,
              })
            }
          />
        </div>
        <div className="field-row">
          <label>url</label>
          <input
            value={draft.publisher?.url ?? ""}
            onChange={(e) =>
              update("publisher", {
                ...(draft.publisher ?? {}),
                url: e.target.value || null,
              })
            }
          />
        </div>
        <div className="field-row">
          <label>contact</label>
          <input
            value={draft.publisher?.contact ?? ""}
            onChange={(e) =>
              update("publisher", {
                ...(draft.publisher ?? {}),
                contact: e.target.value || null,
              })
            }
          />
        </div>
      </section>
    </div>
  );
}

/**
 * Raw-JSON textarea that parses on every keystroke. Invalid JSON shows a
 * red border and DOESN'T propagate to the draft — the operator sees their
 * draft value but the parent's `onChange` only fires on a successful parse.
 * Empty input → null.
 */
function RawJsonField({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: unknown;
  placeholder: string;
  onChange: (parsed: unknown) => void;
}): JSX.Element {
  const display = value === null || value === undefined ? "" : JSON.stringify(value, null, 2);
  return (
    <div className="field-row">
      <label>{label}</label>
      <textarea
        defaultValue={display}
        placeholder={placeholder}
        onBlur={(e) => {
          const text = e.target.value.trim();
          if (text === "") {
            onChange(null);
            return;
          }
          try {
            onChange(JSON.parse(text));
            e.target.style.borderColor = "";
          } catch {
            e.target.style.borderColor = "var(--danger)";
          }
        }}
      />
    </div>
  );
}
