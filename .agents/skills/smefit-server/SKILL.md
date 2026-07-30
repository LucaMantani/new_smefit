---
name: smefit-server
description: Use this skill when working on the SMEFiT server infrastructure — adding new CLI commands, new metadata fields, new resource types, changing how the bin/trash/delete flow works, or extending the registry. Also use when debugging upload/download/rename/move operations or WebDAV client interactions.
---

# SMEFiT Server Implementation Guide

## Key files

- `smefit/server_utils.py` — all server logic: WebDAV client, registry read/write, upload/download/rename/trash
- `smefit/scripts/smefit_rm.py` — calls `trash()` (moves to `bin/`, removes from registry)
- `smefit/scripts/smefit_mv.py` — calls `rename()`
- `smefit/scripts/smefit_upload.py` — calls `Uploader.upload()`
- `smefit/scripts/smefit_get.py` — calls `Downloader.download()`
- `smefit/scripts/smefit_ls.py` — displays registry via `_fit_rows` / `_report_rows`
- `smefit/scripts/smefit_server.py` — server admin: credentials setup, storage check, sync registry, tutorial
- `smefit/scripts/smefit_manage_project.py` — project list CRUD via `add_project` / `rename_project` / `remove_project`
- `SERVER.md` — user-facing documentation

## CLI commands (entry points in `pyproject.toml`)

| Command | Description |
|---|---|
| `smefit_ls` | List fits/reports/rge/misc from the registry |
| `smefit_upload` | Upload a resource; prompts for comment then project |
| `smefit_get` | Download a resource |
| `smefit_mv` | Rename a resource |
| `smefit_rm` | Move a resource to `bin/` on the server (soft delete) |
| `smefit_restore` | Restore a resource from `bin/` back to its original location |
| `smefit_manage_project` | Manage the project list (add/rename/remove/list) |
| `smefit_server` | Server management: setup credentials, check storage, sync registry, tutorial |
| `smefit_setup_local` | Interactive local setup: writes `.config/paths.yaml`, offers to clone `smefit_database` |
| `smefit_setup_server` | Configure server credentials (`~/.config/smefit/server.yaml`), from a shared YAML or interactively |
| `smefit_sync_registry` | Rebuild `registry.json` from scratch (preserving `projects`) |
| `smefit_mkdir` | Create a directory under `misc/` |
| `view_report` | Download + open a report in the browser |

## Remote layout

```
fits/
  <name>.tar.gz
  rge_matrices/<name>.pkl      # extracted from fit archive on upload
  runcards/<name>.yaml          # extracted from fit archive on upload
reports/
  <name>.tar.gz
misc/
  registry_misc.json
  <any path>/...
bin/
  fits/<name>.tar.gz            # trashed fits
  fits/rge_matrices/<name>.pkl
  fits/runcards/<name>.yaml
  reports/<name>.tar.gz
  misc/<any path>/...
  registry_bin.json             # deletion log
registry.json                   # fits + reports + projects
```

## Registry structure (registry.json)

```json
{
  "fits": {
    "<name>": {
      "created_at": "ISO-8601",
      "uploaded_by": "name",
      "has_rge": true,
      "rge_path": "rel/path/to/rge_matrix.pkl",
      "runcard_path": "rel/path",
      "comment": "optional",
      "project": "optional"
    }
  },
  "reports": {
    "<name>": { "created_at", "uploaded_by", "comment", "project" }
  },
  "projects": ["project_a", "project_b"]
}
```

Misc resources have a separate `misc/registry_misc.json` with `{path: {uploaded_at, uploaded_by, comment, project}}`.

## Constants to know

```python
RESOURCE_TYPES = ["fit", "report", "misc"]
_ARCHIVABLE_TYPES = ["fit", "report"]
BIN_DIR = "bin"
BIN_REGISTRY_PATH = "bin/registry_bin.json"
REGISTRY_PATH = "registry.json"
MISC_REGISTRY_PATH = "misc/registry_misc.json"
RGE_MATRICES_REMOTE_DIR = "fits/rge_matrices"
RUNCARDS_REMOTE_DIR = "fits/runcards"
```

## Bin registry structure (bin/registry_bin.json)

```json
{
  "fit/my_fit": {
    "deleted_at": "ISO-8601",
    "deleted_by": "name",
    "comment": "optional",
    "resource_type": "fit",
    "resource_name": "my_fit",
    "original_meta": { ... original registry entry ... }
  }
}
```

Key is `"<resource_type>/<resource_name>"`. `_read_bin_registry` / `_write_bin_registry` are the helpers; `Downloader.get_bin_registry()` exposes it for read-only use. `smefit_ls bin` displays it via `_bin_rows()`.

## Design rules

1. **Two servers**: `public` (bundled read-only creds; team members can have write creds) and `private`. Pass `server=None` to auto-detect.
2. **Registry is atomic**: always `_read_registry` → mutate → `_write_registry` in one function (they download/upload the JSON via a temp file). Never partial writes.
3. **`_ensure_remote_path`**: call before any `client.move()` or `client.upload_sync()` to a new path.
4. **`_empty_registry()`** always includes `"projects": []`. Migration from old flat format happens in `_read_registry`.
5. **`trash()` not `delete()`**: `smefit_rm` moves to `bin/` to avoid data loss.
6. **Fit sub-files**: rge and runcard are stored separately and must be moved/deleted/trashed alongside the main archive.
7. **`sync_registry` rebuilds from scratch** but must preserve the managed lists — `projects` survives via the `_empty_registry` merge. Any new managed list needs the same treatment.
8. **`Uploader.upload()` takes an optional `project` kwarg**, stored in the registry entry.

## How to add a new metadata field

1. Add to the `entry` dict in `Uploader.upload()`.
2. Add to `sync_registry()` if it should survive a resync (preserve from old registry).
3. Update `_fit_rows` / `_report_rows` in `smefit_ls.py` — show column only when at least one resource has it set.
4. Document in `SERVER.md`.

## How to add a new CLI command

1. Create `smefit/scripts/smefit_<name>.py` with a `main()` entry point.
2. Register in `pyproject.toml` under `[project.scripts]`.
3. Import server functions from `smefit.server_utils`.
4. Add to the CLI command table above and to `SERVER.md`.
5. **Update `_COMMANDS` in `smefit/scripts/smefit_server.py`** — the tutorial must stay in sync with the actual command set and descriptions. Keep descriptions accurate (e.g. "Move to bin" not "Delete"). Continuation lines in multi-line descriptions use a plain `\n` with no extra leading spaces; `_print_tutorial` handles alignment automatically via `cmd_w + 10`.

## WebDAV client cheatsheet

```python
client.check(path)  # exists?
client.list(path)  # list dir (returns name strings including dir itself)
client.mkdir(path)  # create dir
client.move(remote_path_from, remote_path_to)
client.clean(path)  # delete
client.upload_sync(remote_path, local_path)
client.download_sync(remote_path, local_path)
client.free()  # free bytes
```

Note: `client.list()` returns bare names (not full paths). Entries include the directory itself and may have trailing `/`. Always strip and filter.

## Common gotchas

- `client.move()` requires the destination *parent* directory to exist — always call `_ensure_remote_path` first.
- `_list_resource_names` skips the directory entry itself and `registry_misc.json`; for archivable types it strips `.tar.gz`.
- `smefit_rm` used to call `delete()` (permanent). It now calls `trash()` (moves to `bin/`). If you need permanent delete, call `delete()` directly from Python.
- Misc resources use a flat path inside `misc/`; their registry key is that full path (e.g. `results/run1/output.pkl`).
