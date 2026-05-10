/**
 * Wire shapes mirroring `codec_supervisor/safety.py` Pydantic models.
 *
 * If you change the Python side, update these too — there's no codegen
 * step yet (slice 5 is a thin admin app, not a polyglot SDK). The PUT
 * endpoint validates the body, so a drift will surface as a 400 / 422
 * rather than silent corruption.
 */

export type CategoryAction = "stop" | "redact" | "regenerate" | "flag";

export interface Category {
  name: string;
  action: CategoryAction;
  description?: string | null;
}

export interface ClassifierBlock {
  family: string;
  host?: "server" | "client" | "both" | null;
  requires_engine_features?: ReadonlyArray<
    "logits_processor" | "hidden_states" | "sampling_chain"
  > | null;
}

export interface ClassifierInternalBlock {
  thresholds?: Record<string, number> | null;
  weights_url?: string | null;
}

export interface ClientHooksBlock {
  prefilter_categories?: readonly string[] | null;
  client_classifier_family?: string | null;
}

export interface PublisherBlock {
  name?: string | null;
  url?: string | null;
  contact?: string | null;
}

export interface InternalPolicy {
  id: string;
  version: string;
  tokenizers: string[];
  categories: Category[];
  category_registry?: string | null;
  classifier: ClassifierBlock;
  client_hooks?: ClientHooksBlock | null;
  published_at?: string | null;
  publisher?: PublisherBlock | null;

  // Internal-only — sanitized away before publish.
  banned_token_ids?: number[] | null;
  regex_patterns?: string[] | null;
  grammar_constraints?: Array<Record<string, unknown>> | null;
  multi_token_patterns?: Array<Record<string, unknown>> | null;
  classifier_internal?: ClassifierInternalBlock | null;
}

export interface RulesSummary {
  banned_token_id_count?: number | null;
  regex_pattern_count?: number | null;
  grammar_constraint_count?: number | null;
  multi_token_pattern_count?: number | null;
}

export interface PublishedDescriptor {
  id: string;
  version: string;
  tokenizers: string[];
  categories: Category[];
  category_registry?: string | null;
  classifier: ClassifierBlock;
  rules_summary?: RulesSummary | null;
  client_hooks?: ClientHooksBlock | null;
  published_at?: string | null;
  publisher?: PublisherBlock | null;
}

export interface SanitizeResponse {
  descriptor: PublishedDescriptor;
  hash: string;
}
