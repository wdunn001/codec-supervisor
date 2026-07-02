# Offline build: codec-metamcp

Everything needed to rebuild codec-metamcp with **no internet**, using lab-local mirrors.

## Inputs (all mirrored on devops.quasarke.net / 192.168.1.88:8530)

| Input | Offline source |
|---|---|
| MetaMCP fork source (`feat/codec-binary-transport`) | git `akadmin/metamcp` (all upstream+fork branches/tags); local checkout `/home/vinez/metamcp`; worktree `./metamcp-src` |
| This repo (build defs) | git `akadmin/codec-supervisor`; local `/home/vinez/codec-supervisor` |
| Codec dicts/maps | git `akadmin/Codec`; dict vendored at `./offline-assets/dicts/` |
| `node:24-bookworm-slim` | registry `mirror/node:24-bookworm-slim` |
| `ghcr.io/astral-sh/uv:debian` (uv/uvx binaries) | registry `mirror/uv:debian` |
| npm deps | Docker layer cache on .88 (populated) + vendored tarballs in `/mnt/data/offline-vendor/` |
| Known-good images | registry `mirror/codec-metamcp:v0.5.0` (running prod), `mirror/codec-metamcp:0634f90-offline`, `mirror/metamcp:2.4.15-local` (upstream) |

## Rebuild (offline)

```sh
cd /home/vinez/codec-supervisor
# update ./metamcp-src to the ref you want (worktree of /home/vinez/metamcp):
git -C metamcp-src checkout <ref>
docker build -f Dockerfile.metamcp.offline -t codec-metamcp:local .
```

`Dockerfile.metamcp.offline` differs from `Dockerfile.metamcp` by: COPY `./metamcp-src`
instead of `git clone` from GitHub; uv/uvx copied from `mirror/uv:debian` instead of the
astral.sh installer; the zstd dict COPYed from `offline-assets/` instead of
raw.githubusercontent.

## Caveats

- `pnpm install` needs the registry **unless** the .88 layer cache holds the install
  layers (it does after any successful build) or you restore `node_modules` from
  `/mnt/data/offline-vendor/metamcp-node_modules-*.tgz`.
- The base stage `apt-get install` also rides the layer cache; a truly from-scratch
  offline host needs the mirrored images above (pull + retag) rather than a full rebuild.
- Worst case, prod redeploys directly from `mirror/codec-metamcp:v0.5.0` with zero build.
