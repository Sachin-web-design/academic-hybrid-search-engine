# Team Setup Guide — Academic Hybrid Search Engine

Follow these steps in order on your own laptop. If you hit an error, check the
**Troubleshooting** section at the bottom first — we already ran into most of
these while setting this up.

## 1. Install Python 3.11+

Download from python.org. **During installation, check the box "Add python.exe
to PATH"** on the very first screen — this saves a lot of pain later.

Verify: open a new Command Prompt/PowerShell and run:
```
python --version
```

## 2. Install Git

Download from git-scm.com, default options are fine.

Verify:
```
git --version
```

First time only, set your identity:
```
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

## 3. Clone the repository

```
git clone https://github.com/Sachin-web-design/academic-hybrid-search-engine.git
cd academic-hybrid-search-engine
```

## 4. Set up a virtual environment and install dependencies

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

This takes a few minutes (sentence-transformers and faiss-cpu are large
downloads). You should see "Successfully installed ..." at the end.

## 5. Build the search index

The repo does NOT include the pre-built index files (they're excluded via
.gitignore since they're large and regenerable). Build them locally:

```
python indexer/build_index.py --input data/pages.jsonl
```

First run downloads an embedding model (~90MB) — needs internet.

## 6. Test it works

```
python -m search.hybrid "neural information retrieval"
```

You should see a ranked list of real paper titles. If so, your setup is
complete.

## 7. Run the full app

Terminal 1:
```
uvicorn api.main:app --reload
```

Then open `frontend/index.html` by double-clicking it in File Explorer —
it will open in your browser and talk to the API automatically.

---

## Troubleshooting (issues we actually hit — check here first)

**`python` is not recognized as a command**
Python wasn't added to PATH during install. Reinstall Python, choose
"Modify", and on the "Advanced Options" screen check "Add Python to
environment variables". Close and reopen your terminal after.

**`pip install -r requirements.txt` says "Could not open requirements
file"**
You're probably in the wrong folder — the zip/repo sometimes extracts into
a nested folder. Run `dir` (Windows) to see what's actually in your current
folder, and `cd` into the right one until `requirements.txt` shows up.

**Import error mentioning "Application Control policy has blocked this
file" (usually during `build_index.py`, from scipy)**
This is Windows **Smart App Control** blocking an unsigned library file.
Go to Windows Security → App & browser control → Smart App Control
settings → turn it **Off** → restart your PC. (Note: once off, turning it
back on requires a Windows reinstall — this is normal for dev machines.)

**`SSLCertVerificationError` / `CERTIFICATE_VERIFY_FAILED` when running
`arxiv_fetcher.py`**
Run:
```
pip install pip-system-certs
```
Then open a **new** terminal and try again.

**Git commit fails with "Please tell me who you are"**
Run the two `git config --global` commands from Step 2 above.

**Warnings like "LF will be replaced by CRLF"**
This is completely normal on Windows and not an error — ignore it.

**Editing `.gitignore` in Notepad doesn't seem to work (ignored files
still get added)**
Notepad can save with the wrong line endings. If this happens, ask
whoever set up the repo (Sachin) rather than debugging it yourself — it's
already fixed in this repo.
