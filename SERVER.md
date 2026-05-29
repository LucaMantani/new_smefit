# SMEFiT Server Commands

Resources are organised into two types: `fit` and `report`.
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
smefit_setup_server my_credentials.yaml

# Or interactively
smefit_setup_server
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
Use `--force` to overwrite an existing config.

---

## List resources on the server

```bash
# Display the full registry (fits and reports with all metadata)
smefit_ls registry

# List fits — shows creation date and RGE status from the registry
smefit_ls fit

# List reports
smefit_ls report

# List fits that have an rge_matrix.pkl (reads from registry, fast)
smefit_ls rge

# Target a specific server
smefit_ls registry --server public
smefit_ls fit --server private
```

`smefit_ls registry` output example:
```
Fits (2):
  my_fit_v1    2026-05-20  rge=yes
  my_fit_v2    2026-05-29  rge=no

Reports (0):
  (none)
```

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

# Explicitly target the public server
smefit_get fit my_fit --server public
```

If `--view` is used and the report is already present locally, it is opened directly without re-downloading.

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

If the report is already present locally it is opened directly without re-downloading.

---

## Rename a resource (team members only)

```bash
smefit_mv fit old_name new_name
smefit_mv report old_name new_name --server public
```

## Delete resources (team members only)

```bash
# Prompts for confirmation
smefit_rm fit my_fit

# Delete multiple resources at once
smefit_rm fit fit_a fit_b fit_c

# Skip confirmation prompt
smefit_rm fit my_fit -f

smefit_rm report my_report --server public
```

## Sync the fit registry (team members only)

The fit registry (`fits/registry.json`) is updated automatically by `smefit_upload`,
`smefit_mv`, and `smefit_rm`. If it ever drifts out of sync (e.g. files moved outside
these tools), rebuild it from scratch:

```bash
smefit_sync_registry
```

For fits, every archive is downloaded to re-detect `has_rge`. For reports, only
names are listed (no download needed). Existing `created_at` and `uploaded_by`
values are preserved where possible; entries not previously in the registry
receive the current time as a fallback. Requires write credentials.

---

## Upload (team members only)

```bash
# Upload a fit (private server by default if configured)
# You will be prompted for an optional comment (press Enter to skip)
smefit_upload fit my_fit

# Pass a comment directly with -m to skip the interactive prompt
smefit_upload fit my_fit -m "baseline run, no RGE"

# Upload from a specific path
smefit_upload fit my_fit /path/to/my_fit

# Overwrite if it already exists
smefit_upload fit my_fit --force

# Upload a report to the public server
smefit_upload report my_report --server public
```

The comment is stored in the registry and shown in `smefit_ls` output next to the resource name.
