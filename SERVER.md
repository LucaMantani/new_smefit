# SMEFiT Server Commands

Resources are organised into three types: `fit`, `report`, and `misc`.
RGE matrices (`rge_matrix.pkl`) live inside fit directories and are not standalone resources.
Two servers are available: `public` and `private`.

**Default server selection** — all commands auto-detect the server:

- Private credentials configured → **private server** used by default.
- No credentials → **public server** (bundled read-only access, no setup needed).

Use `--server public` or `--server private` to override explicitly.

---

## First-time setup (team members only)

External users get read-only access to the public server automatically — no setup needed.

Team members must run the setup once to configure their credentials:

```bash
# From a privately-shared credentials file (recommended)
smefit_server setup my_credentials.yaml

# Or interactively
smefit_server setup

# Overwrite an existing config
smefit_server setup my_credentials.yaml --force
```

The credentials file format (include only the profiles you have access to):

```yaml
public:
  webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
  webdav_login:    <write-token>
  webdav_password: <password>
  name:            Alice  # optional — shown in the registry next to uploaded fits
private:
  webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
  webdav_login:    <private-token>
  webdav_password: <password>
  name:            Alice
```

Credentials are saved to `~/.config/smefit/server.yaml` (mode 600).

---

## List resources on the server

```bash
# Display the full registry (fits, reports, projects)
smefit_ls
smefit_ls registry

# List fits — shows creation date, RGE status, uploader
smefit_ls fit

# List reports
smefit_ls report

# List fits that have an rge_matrix.pkl (reads from registry, fast)
smefit_ls rge

# List contents of misc/ with metadata
smefit_ls misc

# List trashed resources with deletion metadata
smefit_ls bin
smefit_ls bin --type fit   # filter by resource type: fit, report, or misc

# Filter fits/reports/registry by project label
smefit_ls fit --project linear_fits

# Target a specific server
smefit_ls registry --server public
```

`smefit_ls registry` output example:

```text
  Fits (2)
  ┌─────────────┬────────────┬─────┬─────────────┐
  │ Name        │ Date       │ RGE │ Uploaded by │
  ├─────────────┼────────────┼─────┼─────────────┤
  │ my_fit_v1   │ 2026-05-20 │ yes │ Alice       │
  │ my_fit_v2   │ 2026-05-29 │ no  │ Bob         │
  └─────────────┴────────────┴─────┴─────────────┘

  Reports (0)
  (none)

  Projects (1)
    linear_fits
```

---

## Upload (team members only)

```bash
# Upload a fit — prompts for an optional comment, then project selection
smefit_upload fit my_fit

# Pass a comment directly with -m to skip the interactive prompt
smefit_upload fit my_fit -m "baseline run, no RGE"

# Upload from a specific local path
smefit_upload fit my_fit /path/to/my_fit

# Overwrite if it already exists on the server
smefit_upload fit my_fit --force

# Upload a report to the public server
smefit_upload report my_report --server public

# Upload a file to misc/
smefit_upload misc output.pkl
smefit_upload misc results/run1/output.pkl /local/path/output.pkl
```

The comment is stored in the registry and shown in `smefit_ls` output.
After the comment prompt, you are shown the project list and can pick one by number
(press Enter to skip). Use `--project NAME` to assign a project non-interactively.

---

## Download

```bash
# Download a fit or report
smefit_get fit my_fit
smefit_get report my_report

# Download to a specific directory
smefit_get fit my_fit /path/to/output

# Download a report and open it in the browser
smefit_get report my_report --view
smefit_get report my_report /path/to/output --view --server public

# Download only the rge_matrix.pkl from a fit
smefit_get rge my_fit
smefit_get rge my_fit /path/to/output

# Download a file from misc/
smefit_get misc results/run1/output.pkl
smefit_get misc results/run1/output.pkl /local/output/dir/

# Explicitly target the public server
smefit_get fit my_fit --server public
```

If `--view` is used and the report is already present locally, it is opened directly
without re-downloading.

---

## View a report (shortcut)

`view_report` is a shortcut for `smefit_get report ... --view`.

```bash
# Download (if needed) and open a report in the browser
view_report my_report

# Save to a specific directory
view_report my_report /path/to/output

# Explicitly target a server
view_report my_report --server public
view_report my_report --server private
```

---

## misc/ — free-form storage (team members only)

The `misc/` folder has no assumed structure. Files can be stored at any path inside it.

```bash
# Create a directory structure inside misc/
smefit_mkdir results/run1
smefit_mkdir matrices/2026/june

# Rename or remove entries in misc/
smefit_mv misc old/path new/path
smefit_rm misc results/run1/output.pkl
```

`smefit_mkdir` only works in `misc/` — it will error if called on fits or reports.

---

## Rename a resource (team members only)

```bash
smefit_mv fit old_name new_name
smefit_mv report old_name new_name --server public
smefit_mv misc old/path new/path
```

---

## Remove resources (team members only)

Resources are **not permanently deleted** — they are moved to a `bin/` folder at the root
of the remote server (mirroring the original path). Associated files (rge matrix, runcard)
are moved alongside the main archive. The resource is removed from the registry and its
entry (date, deleter, comment, original metadata) is written to `bin/registry_bin.json`.

```bash
# Prompts for a deletion comment, then confirmation
smefit_rm fit my_fit

# Provide comment inline to skip the prompt
smefit_rm fit my_fit -m "superseded by v2"

# Move multiple resources to bin at once (same comment applied to all)
smefit_rm fit fit_a fit_b fit_c

# Skip confirmation prompt
smefit_rm fit my_fit -f

smefit_rm report my_report --server public
smefit_rm misc results/run1/output.pkl
```

View what is currently in the bin:

```bash
smefit_ls bin
smefit_ls bin --type fit
```

---

## Restore resources (team members only)

Moves a resource from `bin/` back to its original remote path and reinstates its
registry entry. Fails if the original location is already occupied.

```bash
smefit_restore fit my_fit
smefit_restore report my_report --server public
smefit_restore misc results/run1/output.pkl
```

---

## Projects (team members only)

Projects are metadata labels that can be attached to fits and reports to group them.
The list of valid projects is stored in the registry and managed with `smefit_manage_project`.

```bash
# List available projects
smefit_manage_project list

# Add a new project
smefit_manage_project add linear_fits

# Rename a project (updates all resources that reference it)
smefit_manage_project rename linear_fits linear_analyses

# Remove a project
smefit_manage_project remove linear_analyses
```

During `smefit_upload`, after the comment prompt, you are shown the project list and
can pick one by number (press Enter to skip). Use `--project NAME` to assign a project
non-interactively:

```bash
smefit_upload fit my_fit --project linear_fits
```

The project label is displayed as a column in `smefit_ls registry` and `smefit_ls fit/report`
whenever at least one resource has a project assigned.

---

## Server management

`smefit_server` groups setup, storage, registry-sync, and the command reference into one place.

```bash
# Configure credentials (from a shared YAML file or interactively)
smefit_server setup my_credentials.yaml
smefit_server setup               # interactive
smefit_server setup --force       # overwrite existing config

# Show used / free space on the server
smefit_server storage
smefit_server storage --server private

# Rebuild the registry from scratch (team members only)
smefit_server sync
smefit_server sync --server public

# List all available smefit commands with descriptions
smefit_server tutorial
```

### Sync details

`registry.json` is updated automatically by `smefit_upload`, `smefit_mv`, and `smefit_rm`.
If it ever drifts out of sync (e.g. files moved outside these tools), rebuild it with
`smefit_server sync`.

For fits, every archive is downloaded to re-detect `has_rge`. For reports, only names are
listed (no download needed). Existing `created_at` and `uploaded_by` values are preserved
where possible; entries not previously in the registry receive the current time as a fallback.
Requires write credentials.
