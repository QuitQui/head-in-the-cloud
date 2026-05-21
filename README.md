# head-in-the-cloud

Run GPU training jobs on Kaggle from your local machine with one command.

```
hitc run train.py
```

`hitc` packs your project, uploads it to Kaggle as a private dataset, launches a GPU kernel, waits for it to finish, and downloads the output — all without you touching the Kaggle UI.

---

## How it works

```
local project
    │
    ▼
packer          ── tar.gz of .py / .yaml / .json / .toml / .txt files
    │               (respects .gpuignore, skips model weights)
    ▼
kaggle_client   ── uploads archive as a private Kaggle dataset
    │               creates a new dataset or bumps the version automatically
    ▼
kaggle_client   ── pushes a kernel that mounts the dataset, runs your script
    │               GPU enabled, internet off, /kaggle/working as cwd
    ▼
kaggle_client   ── polls kernels_status() until complete / error / cancelled
    │
    ▼
collector       ── downloads output files, zips them as results_YYYYMMDD_HHMMSS.zip
    │
    ▼
notifier        ── sends a Line / Slack message via `link send`
```

---

## Prerequisites

**Kaggle API credentials** — create `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

Get your key at kaggle.com → Account → API → Create New Token.

**Python 3.10+** and [uv](https://docs.astral.sh/uv/).

**[Link](https://github.com/QuitQui/Link)** (optional) — for notifications. If `link` is not installed, the notify step is silently skipped.

---

## Installation

```bash
git clone https://github.com/QuitQui/head-in-the-cloud
cd head-in-the-cloud
uv sync
```

---

## Quick start

```bash
# Run train.py from the current directory on a Kaggle GPU
hitc run train.py

# Point at a specific project directory
hitc run train.py --dir /path/to/project

# Use a custom dataset slug and kernel slug
hitc run train.py --dataset my-dataset --kernel my-kernel
```

Output is saved to `./output/results_<timestamp>.zip` in your working directory.

---

## Ignoring files

Create a `.gpuignore` in your project root. Syntax is identical to `.gitignore`:

```
# always excluded automatically:
#   __pycache__/, *.pyc, .venv/, venv/
#   *.pt, *.pth, *.ckpt, *.bin, *.safetensors  (weight files > 100 MB)

# add your own:
data/raw/
notebooks/
secrets.env
```

Only these extensions are packed: `.py`, `.yaml`, `.yml`, `.json`, `.toml`, `.txt`.

---

## Module reference

| Module | Role |
|---|---|
| `headinthecloud.packer` | `pack(project_dir, ignore_file)` → `Path` (tar.gz) |
| `headinthecloud.kaggle_client` | `upload_dataset`, `run_kernel`, `poll_kernel`, `download_output` |
| `headinthecloud.collector` | `collect(kernel_output_dir, output_dir)` → `Path` (zip) |
| `headinthecloud.notifier` | `notify(message)` — shells out to `link send` |

---

## Development

```bash
uv sync
uv run pytest          # 37 tests across 4 modules
```

### Phases

| Phase | Branch | What shipped |
|---|---|---|
| 1 — Scaffold | `feat/scaffold` | Project layout, pyproject.toml, CI skeleton |
| 2A — Packer | `feat/packer` | `packer.py` + 10 tests |
| 2B — Kaggle client | `feat/kaggle-client` | `kaggle_client.py` + 18 tests |
| 2C — Collector + Notifier | `feat/collector-notifier` | `collector.py`, `notifier.py` + 9 tests |

---

## License

MIT
