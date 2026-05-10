/**
 * Thin REST client for codec-supervisor's `/admin/policies/*` endpoints.
 *
 * Same-origin in production (the admin SPA is served by the supervisor at
 * `/admin/policies/`); cross-origin during `npm run dev` via the Vite proxy
 * configured in `vite.config.ts`.
 */
import type { InternalPolicy, PublishedDescriptor, SanitizeResponse } from "./types";

const BASE = "/admin/policies";

class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);
  if (!res.ok) {
    let detail: string;
    try {
      const body = await res.json();
      detail = (body as { detail?: string }).detail ?? `HTTP ${res.status}`;
    } catch {
      detail = `HTTP ${res.status}`;
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export async function listPolicies(): Promise<string[]> {
  const r = await request<{ policies: string[] }>(BASE);
  return r.policies;
}

export async function getPolicy(id: string): Promise<InternalPolicy> {
  return request<InternalPolicy>(`${BASE}/${encodePolicyId(id)}`);
}

export async function putPolicy(id: string, body: InternalPolicy): Promise<SanitizeResponse> {
  return request<SanitizeResponse>(`${BASE}/${encodePolicyId(id)}`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function deletePolicy(id: string): Promise<void> {
  await request<{ deleted: string }>(`${BASE}/${encodePolicyId(id)}`, {
    method: "DELETE",
  });
}

export async function sanitizePolicy(id: string): Promise<SanitizeResponse> {
  return request<SanitizeResponse>(`${BASE}/${encodePolicyId(id)}/sanitize`, {
    method: "POST",
  });
}

export async function listVersions(): Promise<string[]> {
  const r = await request<{ versions: string[] }>(`${BASE}/_versions`);
  return r.versions;
}

export async function getVersion(hexHash: string): Promise<PublishedDescriptor> {
  return request<PublishedDescriptor>(`${BASE}/_versions/${hexHash}`);
}

export { ApiError };

/**
 * Policy ids contain `/` (e.g. `acme/strict-v3`) which the FastAPI router
 * accepts as a `:path` parameter. Slashes are preserved verbatim — only
 * encode characters that would change URL semantics.
 */
function encodePolicyId(id: string): string {
  return id
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}
