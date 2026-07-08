import os
import hashlib
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
import lancedb
import ollama
import tiktoken

from .parser_utils import get_structural_chunks, fallback_line_chunker

mcp = FastMCP("Multi-Repo Indexer")

# Vector Database Path Setup
DB_DIR = os.path.expanduser("~/.local_multi_repo_mcp_db")
db = lancedb.connect(DB_DIR)

# Initialize a global thread-safe tokenizer (using gpt-4o/cl100k layout standards)
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = tiktoken.get_encoding("gpt-4")

def count_tokens(text: str) -> int:
    """Computes exact token weight locally using standard byte pair encoding."""
    return len(TOKENIZER.encode(text, disallowed_special=()))

def get_embedding(text: str) -> List[float]:
    """Generates a local 768-dimension embedding via Ollama."""
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Ollama connection error. Ensure 'ollama pull nomic-embed-text' was run. Details: {e}")

def get_clean_table_name(repo_path: str) -> str:
    """Generates a unique, alphanumeric safe table name from a file directory path."""
    repo_name = os.path.basename(os.path.normpath(repo_path)).replace("-", "_").replace(".", "_")
    path_hash = hashlib.md5(repo_path.encode()).hexdigest()[:8]
    return f"repo_{repo_name}_{path_hash}"

def update_manifest(repo_path: str, table_name: str, elements_count: int):
    """Saves tracked repositories into a global schema index for AI search models."""
    manifest_data = [{
        "repo_path": os.path.abspath(repo_path),
        "repo_name": os.path.basename(os.path.normpath(repo_path)),
        "table_name": table_name,
        "total_chunks": elements_count
    }]
    if "master_manifest" in db.table_names():
        tbl = db.open_table("master_manifest")
        tbl.delete(f"repo_path = '{os.path.abspath(repo_path)}'")
        tbl.add(manifest_data)
    else:
        # FIX: Added mode="overwrite" to force LanceDB to cleanly 
        # initialize the master tracking layout file safely without conflicts.
        db.create_table("master_manifest", data=manifest_data, mode="overwrite")

@mcp.tool()
def list_indexed_repositories() -> str:
    """Lists all active codebase repositories registered inside the system database."""
    if "master_manifest" not in db.table_names():
        return "No repositories have been indexed yet."
    
    tbl = db.open_table("master_manifest")
    
    # FIX: You cannot call .to_list() on a raw table object. 
    # Calling .search() without args starts a complete scanning scan that supports .to_list()
    records = tbl.search().to_list()
    
    if not records:
        return "No repositories found in manifest tracker."
        
    output = ["=== Indexed Codebases ==="]
    for r in records:
        output.append(f"- Name: {r['repo_name']}\n  Path: {r['repo_path']}\n  Chunks: {r['total_chunks']}\n")
    return "\n".join(output)

@mcp.tool()
def index_repository(repo_path: str) -> str:
    """Indexes a repository folder with verbose terminal debugging logs."""
    abs_path = os.path.abspath(repo_path)
    if not os.path.exists(abs_path):
        return f"Error: Path {abs_path} does not exist."
        
    table_name = get_clean_table_name(abs_path)
    batch_records = []
    
    supported_extensions = ('.js', '.jsx', '.ts', '.tsx', '.java', '.py')
    ignore_dirs = {
        '.git', '.github', 'node_modules', '__pycache__', 'venv', '.venv', 'env', 
        'dist', 'build', '.idea', '.vscode', 'out', 'target', 'bin', 'obj',
        'coverage', '.nyc_output', 'public', 'vendor', 'pods', '.poetry', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', '.tox', '.eggs', '.cache', 'site-packages'
    }
    # Suffix-based directory exclusions (e.g. '*.egg-info', '.*.egg-info')
    ignore_dir_suffixes = ('.egg-info',)
    ignore_files = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'cargo.lock', 'go.sum',
        'poetry.lock', 'poetry.toml', 'pipfile', 'pipfile.lock', '.env', 'tsconfig.json'
    }

    # REMOVED EMOJIS TO PREVENT WINDOWS CHARMAP ERRORS
    print(f"\n[DEBUG] Starting walk on: {abs_path}")
    
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [
            d for d in dirs
            if d not in ignore_dirs
            and not d.startswith('.')
            and not d.endswith(ignore_dir_suffixes)
        ]
        print(f"[DEBUG] Scanning directory: {root}")
        
        for file in files:
            if file in ignore_files or file.startswith('.env'):
                continue
                
            if file.endswith(supported_extensions):
                full_file_path = os.path.normpath(os.path.join(root, file))
                print(f"[DEBUG] Found file matching extension: {file}")
                
                try:
                    file_size = os.path.getsize(full_file_path)
                    print(f"   [DEBUG] File Size: {file_size} bytes")
                    
                    if file_size > 2 * 1024 * 1024:
                        print("   [DEBUG] Skipped: File exceeds 2MB limit.")
                        continue
                        
                    with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    chunks = get_structural_chunks(full_file_path, content)
                    print(f"   [DEBUG] Tree-sitter chunks found: {len(chunks)}")
                    
                    if not chunks and content.strip():
                        print("   [DEBUG] Applying fallback_line_chunker to flat script...")
                        chunks = fallback_line_chunker(full_file_path, content, max_lines=80)
                        print(f"   [DEBUG] Fallback chunks created: {len(chunks)}")
                        
                    for chunk in chunks:
                        chunk["vector"] = get_embedding(chunk["text"])
                        batch_records.append(chunk)
                        
                except Exception as e:
                    print(f"   [DEBUG] Error processing file {file}: {str(e)}")
                    continue

    print(f"[DEBUG] Total records prepared for database: {len(batch_records)}\n")

    if not batch_records:
        return "No processable source files found."

    if table_name in db.table_names():
        db.drop_table(table_name)
        
    table = db.create_table(table_name, data=batch_records)
    
    if len(table) > 300:
        table.create_index(metric="cosine", num_partitions=16, num_sub_vectors=96)
        index_type = "with structural IVF-PQ acceleration"
    else:
        index_type = "with basic vector direct lookup"

    update_manifest(abs_path, table_name, len(batch_records))
    return f"Success: Codebase '{os.path.basename(abs_path)}' indexed completely ({len(batch_records)} nodes {index_type})."

@mcp.tool()
def search_codebase(
    repo_path: str, 
    query: str, 
    file_filter: Optional[str] = None, 
    limit: int = 15, 
    token_budget: int = 4000,
    verbose: bool = True
) -> str:
    """Semantically queries code blocks from an isolated indexed codebase path with a token ceiling safeguard.

    When ``verbose`` is False, each result emits a compact location line plus a
    short preview (~80 chars) instead of the full source block, so the output
    stays scannable for quick lookup.
    """
    target_table = get_clean_table_name(os.path.abspath(repo_path))
    
    if target_table not in db.table_names():
        return f"Error: Repository at '{repo_path}' is not indexed yet. Use index_repository first."
        
    table = db.open_table(target_table)
    query_vector = get_embedding(query)
    
    search_query = table.search(query_vector).metric("cosine")
    if file_filter:
        search_query = search_query.where(f"file_path LIKE '%{file_filter}%'", prefilter=True)
        
    results = search_query.limit(limit).to_list()
    if not results:
        return "No relevant structural definitions found for this query."
        
    output = [f"=== Results for Codebase '{os.path.basename(repo_path)}' ==="]
    current_tokens = count_tokens(output[0])
    truncated = False
    
    for idx, res in enumerate(results):
        if verbose:
            block_text = (
                f"\n[{idx+1}] File: {res['file_path']} (Line {res['start_line']}) | Type: {res['type']}\n"
                f"Code:\n{res['text']}\n" + "-"*40
            )
        else:
            preview = res['text'].replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            block_text = (
                f"\n[{idx+1}] File: {res['file_path']} (Line {res['start_line']}) | Type: {res['type']} | Preview: {preview}"
            )
        block_tokens = count_tokens(block_text)
        
        # Enforce Token Safeguard Guardrails
        if current_tokens + block_tokens > token_budget:
            truncated = True
            break
            
        output.append(block_text)
        current_tokens += block_tokens
        
    if truncated:
        output.append(f"\n⚠️ WARNING: Search results truncated to fit within the {token_budget} token context budget.")
        
    return "\n".join(output)

@mcp.tool()
def delete_repository(repo_path_or_table: str) -> str:
    """
    Removes a repository completely from LanceDB using either its 
    system directory path OR its explicit database table name.
    """
    # Force absolute paths if a path structure is given
    if os.path.exists(repo_path_or_table) or "/" in repo_path_or_table or "\\" in repo_path_or_table:
        abs_path = os.path.abspath(repo_path_or_table)
        table_name = get_clean_table_name(abs_path)
    else:
        # It is a direct table string (e.g., 'repo_local_code_mcp_39e95ad4')
        table_name = repo_path_or_table

    # Standardize on modern lancedb API calls
    all_tables = db.table_names()
    
    table_removed = False
    if table_name in all_tables:
        db.drop_table(table_name)
        table_removed = True
        
    manifest_removed = False
    if "master_manifest" in all_tables:
        try:
            tbl = db.open_table("master_manifest")
            if hasattr(tbl, 'delete'):
                if table_name.startswith("repo_"):
                    tbl.delete(f"table_name = '{table_name}'")
                else:
                    tbl.delete(f"repo_path = '{os.path.abspath(repo_path_or_table)}'")
                manifest_removed = True
        except Exception:
            pass
        
    if table_removed or manifest_removed:
        return f"Successfully pruned and deleted vector data for: {table_name}"
    return f"Identifier '{repo_path_or_table}' (resolved as '{table_name}') was not found in the database."


@mcp.tool()
def search_all_codebases(
    query: str, 
    limit_per_repo: int = 3, 
    token_budget: int = 6000,
    verbose: bool = True
) -> str:
    """Executes a high-speed cross-query across all indexed tables simultaneously with a global token limit.

    When ``verbose`` is False, each result emits a compact location line plus a
    short preview (~80 chars) instead of the full source block, so the output
    stays scannable for quick lookup.
    """
    # Get all active physical table names currently inside your LanceDB directory
    all_table_names = db.table_names()
    
    # Filter down to only tables that hold repository codes (skipping master_manifest)
    repo_tables = [t for t in all_table_names if t.startswith("repo_")]
    
    if not repo_tables:
        return "No repositories have been indexed globally yet."
        
    query_vector = get_embedding(query)
    combined_results = []
    
    # Loop over the actual physical tables directly, bypassing manifest path strings
    for tbl_name in repo_tables:
        try:
            # Clean up the display name for the console layout
            display_name = tbl_name.replace("repo_", "").split("_")[0]
            
            table = db.open_table(tbl_name)
            hits = table.search(query_vector).metric("cosine").limit(limit_per_repo).to_list()
            
            for hit in hits:
                hit['_repo_name'] = display_name
                combined_results.append(hit)
        except Exception as e:
            continue
            
    if not combined_results:
        return "No structural components matched across any code bases."
        
    # Order results globally based on cosine matrix distances
    combined_results.sort(key=lambda x: x.get('_distance', 1.0))
    
    output = [f"=== Cross-Repo Search Results for: '{query}' ==="]
    
    # Convert list header to clean string format for tokenizer tracking
    current_tokens = count_tokens("\n".join(output))
    truncated = False
    
    for idx, res in enumerate(combined_results):
        dist = res.get('_distance', 0.0)
        if verbose:
            block_text = (
                f"\n[{idx+1}] [Repo: {res['_repo_name']}] | File: {res['file_path']} (Line {res['start_line']}) [Distance: {dist:.4f}]\n"
                f"Code:\n{res['text']}\n" + "-"*40
            )
        else:
            preview = res['text'].replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            block_text = (
                f"\n[{idx+1}] [Repo: {res['_repo_name']}] | File: {res['file_path']} (Line {res['start_line']}) [Dist: {dist:.4f}] | Preview: {preview}"
            )
        block_tokens = count_tokens(block_text)
        
        # Enforce Token Safeguard Guardrails
        if current_tokens + block_tokens > token_budget:
            truncated = True
            break
            
        output.append(block_text)
        current_tokens += block_tokens
        
    if truncated:
        output.append(f"\n⚠️ WARNING: Global cross-repo results truncated to fit within the {token_budget} token context budget.")
        
    return "\n".join(output)

if __name__ == "__main__":
    mcp.run()
