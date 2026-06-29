## Overview
This is a Model Context Protocol (MCP) server that indexes local codebases for semantic search using vector embeddings. It enables AI-assisted code navigation by creating searchable indexes of repositories.

## Core Components

### 1. Project Configuration (pyproject.toml)
- **Project**: `local-code-index` v0.1.0
- **Dependencies**:
  - `mcp[cli]`: Model Context Protocol framework
  - `lancedb`: Vector database for storage
  - `ollama`: Local LLM embedding generation
  - `tree-sitter` + language parsers: Syntax-aware code parsing
  - `tiktoken`: Token counting for LLM context safety

### 2. Main Server Logic (server.py)
**Key Features**:
- Uses FastMCP for MCP server implementation
- LanceDB for vector storage (~/.local_multi_repo_mcp_db)
- Ollama with `nomic-embed-text` model for 768-dimension embeddings
- Token-aware processing using tiktoken (cl100k_base encoding)

**Tools Provided**:
1. `list_indexed_repositories()` - Shows all indexed codebases
2. `index_repository(repo_path)` - Indexes a new codebase
3. `search_codebase(repo_path, query, ...)` - Searches within one codebase
4. `delete_repository(repo_path_or_table)` - Removes an index
5. `search_all_codebases(query, ...)` - Cross-repository search

### 3. Parsing Utilities (parser_utils.py)
**Tree-sitter Based Parsing**:
- Supports JS/TS, Java, Python
- Extracts structural elements: classes, functions, methods, etc.
- Preserves context (comments, decorators)
- Includes test framework detection (Jest, Mocha patterns)
- Fallback to line-based chunking when parsing fails

## Detailed Workflow

### Indexing Process (`index_repository`)
1. **Path Validation**: Converts to absolute path, checks existence
2. **Table Naming**: Creates unique LanceDB table from repo path hash
3. **Directory Walk**:
   - Skips ignored directories (.git, `node_modules`, .venv, `venv`, etc.)
   - Processes only supported extensions (`.js`, `.ts`, `.java`, `.py`)
4. **File Processing**:
   - Reads file content (UTF-8, error-tolerant)
   - Uses tree-sitter to parse into AST
   - Extracts structural chunks with context:
     * Looks back for comments/decorators
     * Respects max_lines limit (80)
     * Marks visited nodes to avoid duplicates
   - Falls back to line chunking if parsing fails
5. **Token Safety**:
   - Counts tokens per chunk using tiktoken
   - Skips chunks exceeding token budget (default 4000)
6. **Vector Storage**:
   - Generates Ollama embeddings (nomic-embed-text)
   - Stores in LanceDB: `[vector, text, file_path, line_number]`
7. **Manifest Update**:
   - Tracks indexed repo in `master_manifest` table
   - Includes path, name, table name, chunk count

### Search Process (`search_codebase`)
1. **Query Embedding**: Converts search text to vector via Ollama
2. **Vector Search**: Finds nearest neighbors in LanceDB table
3. **Token Budgeting**:
   - Accumulates tokens from results
   - Stops when adding next result would exceed budget
4. **Result Formatting**:
   - Returns file path, line numbers, and code snippets
   - Includes chunk type (function/class/etc.)

## Key Technical Details

### Token Safety Mechanisms
- Uses `cl100k_base` encoding (GPT-4 compatible)
- Tracks tokens at chunk level during indexing
- Enforces token budgets during search

Here's a detailed explanation of the codebase for software engineers:

## Overview
This is a Model Context Protocol (MCP) server that indexes local codebases for semantic search using vector embeddings. It enables AI-assisted code navigation by creating searchable indexes of repositories.

## Core Components

### 1. Project Configuration (pyproject.toml)
- **Project**: `local-code-index` v0.1.0
- **Dependencies**:
  - `mcp[cli]`: Model Context Protocol framework
  - `lancedb`: Vector database for storage
  - `ollama`: Local LLM embedding generation
  - `tree-sitter` + language parsers: Syntax-aware code parsing
  - `tiktoken`: Token counting for LLM context safety

### 2. Main Server Logic (server.py)
**Key Features**:
- Uses FastMCP for MCP server implementation
- LanceDB for vector storage (~/.local_multi_repo_mcp_db)
- Ollama with `nomic-embed-text` model for 768-dimension embeddings
- Token-aware processing using tiktoken (cl100k_base encoding)

**Tools Provided**:
1. `list_indexed_repositories()` - Shows all indexed codebases
2. `index_repository(repo_path)` - Indexes a new codebase
3. `search_codebase(repo_path, query, ...)` - Searches within one codebase
4. `delete_repository(repo_path_or_table)` - Removes an index
5. `search_all_codebases(query, ...)` - Cross-repository search

### 3. Parsing Utilities (parser_utils.py)
**Tree-sitter Based Parsing**:
- Supports JS/TS, Java, Python
- Extracts structural elements: classes, functions, methods, etc.
- Preserves context (comments, decorators)
- Includes test framework detection (Jest, Mocha patterns)
- Fallback to line-based chunking when parsing fails

## Detailed Workflow

### Indexing Process (`index_repository`)
1. **Path Validation**: Converts to absolute path, checks existence
2. **Table Naming**: Creates unique LanceDB table from repo path hash
3. **Directory Walk**:
   - Skips ignored directories (.git, `node_modules`, .venv, `venv`, etc.)
   - Processes only supported extensions (`.js`, `.ts`, `.java`, `.py`)
4. **File Processing**:
   - Reads file content (UTF-8, error-tolerant)
   - Uses tree-sitter to parse into AST
   - Extracts structural chunks with context:
     * Looks back for comments/decorators
     * Respects max_lines limit (80)
     * Marks visited nodes to avoid duplicates
   - Falls back to line chunking when parsing fails
5. **Token Safety**:
   - Counts tokens per chunk using tiktoken
   - Skips chunks exceeding token budget (default 4000)
6. **Vector Storage**:
   - Generates Ollama embeddings (nomic-embed-text)
   - Stores in LanceDB: `[vector, text, file_path, line_number]`
7. **Manifest Update**:
   - Tracks indexed repo in `master_manifest` table
   - Includes path, name, table name, chunk count

### Search Process (`search_codebase`)
1. **Query Embedding**: Converts search text to vector via Ollama
2. **Vector Search**: Finds nearest neighbors in LanceDB table
3. **Token Budgeting**:
   - Accumulates tokens from results
   - Stops when adding next result would exceed budget
4. **Result Formatting**:
   - Returns file path, line numbers, and code snippets
   - Includes chunk type (function/class/etc.)

## Key Technical Details

### Token Safety Mechanisms
- Uses `cl100k_base` encoding (GPT-4 compatible)
- Tracks tokens at chunk level during indexing
- Enforces token budgets during search (default 4000 per repo, 6000 for cross-repo)
- Prevents context window overflow when feeding results to LLMs

### Parsing Approach
- **Tree-sitter v0.23+**: Uses modern point tuples (row,column) for precise positioning
- **Context Preservation**: Walks sibling nodes to capture comments/decorators
- **Test Framework Awareness**: Filters out non-test calls in test files
- **Robust Error Handling**: Graceful fallbacks when parsers fail

### Ignored Patterns
Directories: .git, .github, `node_modules`, __pycache__, `venv`, .venv, `env`, `dist`, build, IDE folders, output dirs
Files: Lock files, config files, environment files

### Database Design
- **Per-repo tables**: Isolated vector indexes for each codebase
- **Master manifest**: Central registry of all indexed repositories
- **LanceDB**: Efficient vector storage with metadata filtering

This system provides developers with semantic code search that understands syntax structure while respecting LLM context limitations through intelligent token budgeting. The tree-sitter integration ensures high-quality code chunks that preserve semantic meaning better than simple text splitting.