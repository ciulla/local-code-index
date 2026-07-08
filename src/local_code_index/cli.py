"""Command-line interface for the Local Code Index management tools.

Replaces the legacy ~/.bashrc (or ~/.zshrc) shell-function shortcuts with a
single, cross-platform executable installed alongside the package:

    uv run local-code-index <command> [args]

Available commands:
    idx     [path]   Index a repository folder (defaults to current directory)
    rm      [path]   Remove a repository from the index (defaults to cwd)
    find    <query>  Semantic search across ALL indexed repositories
    search  <path> <query>  Semantic search within ONE indexed repository
    list             List all currently indexed repositories
"""

import argparse
import os
import sys

from . import server


def _resolve_path(path: str | None) -> str:
    """Return an absolute path for the given path (or cwd when None)."""
    target = path if path else "."
    return os.path.abspath(target)


def _cmd_index(args: argparse.Namespace) -> int:
    abs_path = _resolve_path(args.path)
    print(f"\nIndexing codebase: {abs_path}\n")
    result = server.index_repository(abs_path)
    print(result)
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    target = args.path or args.table
    if not target:
        target = "."
    # Only resolve to absolute path if it looks like a filesystem path
    if os.path.exists(target) or "/" in target or "\\" in target:
        target = os.path.abspath(target)
    print(f"\nRemoving vector entries for: {target}\n")
    result = server.delete_repository(target)
    print(result)
    return 0


def _cmd_find(args: argparse.Namespace) -> int:
    if not args.query:
        print("Error: a search query is required. Usage: local-code-index find <query>")
        return 2
    query = " ".join(args.query)
    print(f"\nSearching ALL indexed codebases for: {query!r}\n")
    result = server.search_all_codebases(
        query,
        limit_per_repo=args.limit_per_repo,
        token_budget=args.token_budget,
    )
    print(result)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.query:
        print("Error: a search query is required. Usage: local-code-index search <path> <query>")
        return 2
    abs_path = _resolve_path(args.path)
    query = " ".join(args.query)
    print(f"\nSearching codebase '{os.path.basename(abs_path)}' for: {query!r}\n")
    result = server.search_codebase(
        abs_path,
        query,
        file_filter=args.file_filter,
        limit=args.limit,
        token_budget=args.token_budget,
    )
    print(result)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    result = server.list_indexed_repositories()
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser and subcommand handlers."""
    parser = argparse.ArgumentParser(
        prog="local-code-index",
        description="Manage a local vector index of your codebases.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p_idx = subparsers.add_parser(
        "idx", help="Index a repository folder (defaults to current directory)."
    )
    p_idx.add_argument("path", nargs="?", default=".", help="Path to repository (default: cwd)")
    p_idx.set_defaults(func=_cmd_index)

    p_rm = subparsers.add_parser(
        "rm", help="Remove a repository from the index (by path or table name)."
    )
    p_rm.add_argument("path", nargs="?", help="Repository path (default: cwd)")
    p_rm.add_argument("--table", help="LanceDB table name instead of a path")
    p_rm.set_defaults(func=_cmd_remove)

    p_find = subparsers.add_parser(
        "find", help="Semantic search across ALL indexed repositories."
    )
    p_find.add_argument("query", nargs="+", help="Natural-language search query")
    p_find.add_argument("--limit-per-repo", type=int, default=3, dest="limit_per_repo",
                        help="Max hits per repository (default: 3)")
    p_find.add_argument("--token-budget", type=int, default=6000, dest="token_budget",
                        help="Global token cap on returned results (default: 6000)")
    p_find.set_defaults(func=_cmd_find)

    p_search = subparsers.add_parser(
        "search", help="Semantic search within ONE indexed repository."
    )
    p_search.add_argument("path", nargs="?", default=".", help="Repository path (default: cwd)")
    p_search.add_argument("query", nargs="+", help="Natural-language search query")
    p_search.add_argument("--file-filter", dest="file_filter", default=None,
                          help="Substring filter on file_path (e.g. 'src/')")
    p_search.add_argument("--limit", type=int, default=15, help="Max hits (default: 15)")
    p_search.add_argument("--token-budget", type=int, default=4000, dest="token_budget",
                          help="Token cap on returned results (default: 4000)")
    p_search.set_defaults(func=_cmd_search)

    p_list = subparsers.add_parser(
        "list", help="List all currently indexed repositories."
    )
    p_list.set_defaults(func=_cmd_list)

    return parser


def main() -> int:
    """CLI entry point registered via [project.scripts] in pyproject.toml."""
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
