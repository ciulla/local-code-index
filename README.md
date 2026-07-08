# Local Code Index - MCP Server 🚀

A high-performance, fully local, open-source **Model Context Protocol (MCP)** server built to index massive codebases into an easily searchable format for AI models.

This tool is explicitly optimized for **TypeScript, JavaScript, and Java**, utilizing official **Tree-sitter AST parsing** to capture complete code structures (classes, methods, decorators, annotations, and Javadoc comments) instead of blind text fragments. It runs entirely on your machine via **Ollama** and **LanceDB**, requiring zero API keys and protecting your intellectual property.

---

## 🛠️ Features

- **AST-Aware Structural Chunking**: Groups methods, classes, and relevant context (like `@Get()` decorators in NestJS or Javadoc strings in Spring Boot) into unified semantic records.
- **Scalable Multi-Repo Architecture**: Automatically provisions isolated database tables per repository. Scale up to 100+ codebases incrementally without performance or query degradation.
- **Cross-Repository Search**: Allows AI models to scan one repository or run a global matrix query across all indexed projects simultaneously.
- **Production-Grade File Filtering**: Automatically skips `node_modules`, build outputs, binaries, lockfiles, and environment secrets (`.env`).
- **Sub-Second Latency**: Automatically compiles localized IVF-PQ vector indexes on larger repositories to keep query speeds under a second.

---

## 🏗️ Architecture Design

```text
┌────────────────────────────────────────────────────────┐
│             Your Massive Codebase (TS, JS, Java)       │
└───────────────────────────┬────────────────────────────┘
                            │ (Tree-sitter AST Parsing)
                            ▼
┌────────────────────────────────────────────────────────┐
│     Semantic Chunks (Functions, Classes, Decorators)   │
└───────────────────────────┬────────────────────────────┘
                            │ (Local Ollama nomic-embed-text)
                            ▼
┌────────────────────────────────────────────────────────┐
│    Isolated LanceDB Tables (repo_A, repo_B, etc.)     │
└───────────────────────────┬────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
      [ search_codebase ]       [ search_all_codebases ]
                  ▲                   ▲
                  └─────────┬─────────┘
                            │ (Model Context Protocol)
                            ▼
┌────────────────────────────────────────────────────────┐
│     Your AI Workspace Environment (Cursor / Cline)     │
└────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

Ensure your local project directory matches this setup:

```text
local-code-index/
├── pyproject.toml             # Pin-point environment and tool configurations
├── src/
│   └── local_code_index/
│       ├── __init__.py
│       ├── cli.py             # Cross-platform CLI (uv run local-code-index ...)
│       ├── parser_utils.py    # Official Tree-sitter AST parsing layer
│       └── server.py          # FastMCP server tool and LanceDB engine
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Start Your Local Embedding Model

Make sure [Ollama](https://ollama.com) is installed and active on your machine, then download the code-optimized embedding vector weights:

```bash
ollama pull nomic-embed-text
```

### 2. Install the Project Package

Navigate to your project directory and run the compilation step using `uv` (or standard pip):

```bash
cd local-code-index
uv pip install -e .
```

### 3. Register the VS Code / Editor Extension

To connect this local tool to your AI chat interface, register it inside your favorite editor extension configurations.

#### For Cursor (`Cursor Settings -> Features -> MCP`)

- **Name**: `local-multi-repo-indexer`
- **Type**: `command`
- **Command**: `uv --directory "/absolute/path/to/local-code-index" run python -m local_code_index.server`

#### For Cline (`cline_mcp_settings.json`)

Add this configuration snippet inside your `mcpServers` settings payload:

```json
{
  "mcpServers": {
    "local-multi-repo-indexer": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/local-code-index",
        "run",
        "python",
        "-m",
        "local_code_index.server"
      ],
      "disabled": false
    }
  }
}
```

---

## 💻 Managing the Index from the Integrated Terminal

Indexing and maintenance are driven by a single built-in CLI command (`local-code-index`) that ships with the package — **no `~/.bashrc` or `~/.zshrc` editing required**, and it works the same on Windows PowerShell, macOS and Linux.

A shorter alias **`lci`** is also installed and points at the exact same CLI, so the two invocations below are interchangeable:

```bash
uv run local-code-index idx .   # full name
uv run lci idx .                # short alias
```

> The CLI entry point is registered automatically via `[project.scripts]` in `pyproject.toml` during the `uv pip install -e .` step from the Quick Start. Run it through `uv` so the project's virtualenv is used:

```bash
uv run lci --help
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `idx [path]` | Index a repository folder (defaults to current directory) | `uv run local-code-index idx .` |
| `rm [path]` | Remove a repository from the index | `uv run local-code-index rm ~/dev/my-api` |
| `list` | List all currently indexed repositories | `uv run local-code-index list` |
| `find <query>` | Semantic search across **all** indexed codebases | `uv run local-code-index find "JwtAuthGuard validation logic"` |
| `search [path] <query>` | Semantic search within **one** indexed repository | `uv run local-code-index search . "webhook signature"` |

Each command accepts `--help` for full flag details (token budgets, limit-per-repo, file filters, etc.), for example:

```bash
uv run local-code-index find --help
# Usage: local-code-index find [-h] [--limit-per-repo N] [--token-budget N] [-v] query ...
```

### Concise vs. Verbose Output (`-v` / `--verbose`)

By default, `find` and `search` print a **concise** response — one line per hit showing `[repo] file (line) type distance | Preview: <first 80 chars>` — so you can quickly locate where something lives without scrolling through full source blocks.

Pass `-v` (or `--verbose`) to switch to the **verbose** response, which prints the complete source block for each hit (the original behavior):

```bash
# concise (default): one scannable line per match
uv run local-code-index find "JwtAuthGuard validation logic"

# verbose: include the full code block for every match
uv run local-code-index find -v "JwtAuthGuard validation logic"
uv run local-code-index search . "webhook signature" --verbose
```

> **Tip:** If you prefer even shorter invocations, you can either (a) use the built-in `lci` alias (`uv run lci idx .`), (b) drop `uv run` entirely and call `lci ...` directly after activating the project virtualenv, or (c) create an alias of your own (`alias idx="uv --directory /path/to/local-code-index run lci"`). The CLI itself needs no special shell setup.

---

## 🔥 Practical Examples & Usage

### Workflow 1: From the Integrated Terminal

Simply step into any repository folder on your system and index it with a single command — no shell profile edits needed:

```bash
cd ~/dev/projects/my-nest-api
uv run local-code-index idx .
# Output: Success: Codebase 'my-nest-api' indexed completely (420 nodes with basic vector direct lookup).
```

If you want to run a quick query across everything you've saved:

```bash
uv run local-code-index find "JwtAuthGuard validation logic"
```

---

### Workflow 2: Conversational Prompts via AI Agents (Cursor / Cline)

Once the server status bar is green inside your editor panel, the underlying LLM gains access to your protocol tools natively. You can now use fluid language to ask complex architectural questions.

#### 💡 Example 1: Isolating Features in a Specific Repo

> **User**: _"Check my `payment-service` repo. Do we have a specific method handling webhook signatures?"_
>
> **AI Interaction**: The model implicitly runs `search_codebase` against your project, targeting keyword vectors. It receives the whole relevant function block and returns a complete synthesis of your webhook logic.

#### 💡 Example 2: Cross-Repository Code Archeology

> **User**: _"I need to implement a data-stream handler in this repo. Scan all our indexed codebases to see if we've written a reuseable utility class for this elsewhere so I can copy its pattern."_
>
> **AI Interaction**: The model triggers `search_all_codebases` to search across your microservices. It highlights a match found under a `shared-java-utils` directory, complete with its accompanying Javadocs.

#### 💡 Example 3: Auditing Active Deployments

> **User**: _"List our indexed repositories and tell me which ones are running on our old database schema patterns."_
>
> **AI Interaction**: The model runs `list_indexed_repositories` to find all your project paths and walks through them to identify outdated code conventions.

---

## 🔒 Security & Performance Exclusions

To safeguard memory, protect confidential keys, and maximize performance, files that match the following attributes are omitted from processing:

1. **Directories Skipped**: `.git`, `.github`, `node_modules`, `dist`, `build`, `.vscode`, `target`, `bin`, `vendor`, `__pycache__`, virtualenvs (`venv`/`.venv`/`env`), cache dirs (`.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `.tox`, `.eggs`, `.cache`), `site-packages`, and any `*.egg-info` directory.
2. **Extensions Tracked**: `.js`, `.jsx`, `.ts`, `.tsx`, `.java`.
3. **Blacklisted Configuration Profiles**: `package-lock.json`, `.env`, `.env.local`, `tsconfig.json`.
4. **Size Caps**: Any source code file larger than **2MB** is automatically skipped to prevent execution bottlenecks.

---

### 💡 How the Safeguard Works

1. **Context Window Safety**: The default token budget for single-repo searches is set to `4000` tokens, and global cross-repo searches are capped at `6000` tokens.
2. **Dynamic Truncation**: When the model requests information, the server maps out the top search matches. If a large block threatens to exceed the remaining budget, the engine stops adding data and appends a clean warning flag (`⚠️ WARNING: Global cross-repo results truncated...`).
3. **Model Autonomy**: Because these limits are exposed as parameterized inputs (`token_budget: int = 6000`), sophisticated AI agents like Cursor or Cline can choose to scale the budget up or down depending on their specific model limits.

---

