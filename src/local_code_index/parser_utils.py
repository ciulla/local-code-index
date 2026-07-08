import os
from typing import List, Dict, Any
from tree_sitter import Language, Parser

def get_language_parser(extension: str):
    """Dynamic runtime importer for official language bindings."""
    try:
        if extension in ('.js', '.jsx'):
            import tree_sitter_javascript as tsjs
            lang = Language(tsjs.language())
            return Parser(lang)
        elif extension in ('.ts', '.tsx'):
            import tree_sitter_typescript as tsts
            lang = Language(tsts.language_typescript())
            return Parser(lang)
        elif extension == '.java':
            import tree_sitter_java as tsjava
            lang = Language(tsjava.language())
            return Parser(lang)
        elif extension == '.py':
            import tree_sitter_python as tspy
            lang = Language(tspy.language())
            return Parser(lang)
    except (ImportError, AttributeError):
        return None
    return None

# Combined structural application and test framework rules
TARGET_NODE_TYPES = {
    'class_declaration', 'interface_declaration', 'method_declaration', 
    'constructor_declaration', 'enum_declaration', 'class_definition', 
    'method_definition', 'function_definition', 'function_declaration', 
    'lexical_declaration', 'arrow_function', 'type_alias_declaration',
    'expression_statement', 'call_expression'
}

def is_valid_test_call(node, content: str) -> bool:
    """Filters out normal function executions, matching only test runner frameworks."""
    if node.type in ('expression_statement', 'call_expression'):
        try:
            node_bytes = bytes(content, "utf8")[node.start_byte:node.end_byte]
            node_text = node_bytes.decode("utf8", errors="ignore").strip()
            return any(node_text.startswith(hook) for hook in ('describe(', 'test(', 'it(', 'expect('))
        except Exception:
            return False
    return True

def get_structural_chunks(file_path: str, content: str, max_lines: int = 80) -> List[Dict[str, Any]]:
    """Parses TS, JS, Java, and Python code via modern tree-sitter point tuple structures."""
    ext = os.path.splitext(file_path)[1].lower()  # Force safe extension string parsing
    parser = get_language_parser(ext)
    
    if not parser:
        return []

    try:
        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node
    except Exception:
        return []

    file_lines = content.splitlines()
    chunks = []
    visited_nodes = set()

    def walk_tree(node):
        if node.type in TARGET_NODE_TYPES and node.id not in visited_nodes:
            if node.type in ('expression_statement', 'call_expression') and not is_valid_test_call(node, content):
                for child in node.children:
                    walk_tree(child)
                return

            # MODERN FIX: tree-sitter v0.23+ start_point/end_point are (row, column) tuples!
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            
            # Look back for annotations, decorators, and comments
            actual_start_row = start_row
            prev_sibling = node.prev_sibling
            for _ in range(3):
                if prev_sibling and prev_sibling.type in ('comment', 'decorator', 'annotation'):
                    actual_start_row = prev_sibling.start_point[0] # Fix tuple referencing here too
                    prev_sibling = prev_sibling.prev_sibling
                else:
                    break
            
            total_lines = end_row - actual_start_row + 1
            
            if total_lines <= max_lines:
                node_text = file_lines[actual_start_row:end_row + 1]
                snippet = "\n".join(node_text)
                
                chunk_kind = "test_block" if "test" in file_path.lower() or node.type in ('expression_statement', 'call_expression') else node.type
                
                chunks.append({
                    "text": f"File: {file_path}\nType: {chunk_kind}\nLines {actual_start_row+1}-{end_row+1}\n\n{snippet}",
                    "file_path": file_path,
                    "start_line": actual_start_row + 1,
                    "type": chunk_kind
                })
                mark_subtree_visited(node)
                return

        for child in node.children:
            walk_tree(child)

    def mark_subtree_visited(n):
        visited_nodes.add(n.id)
        for child in n.children:
            mark_subtree_visited(child)

    walk_tree(root_node)
    return chunks

def fallback_line_chunker(file_path: str, content: str, max_lines: int) -> List[Dict[str, Any]]:
    chunks = []
    lines = content.splitlines()
    for i in range(0, len(lines), max_lines - 15):
        chunk_lines = lines[i:i + max_lines]
        if not chunk_lines:
            break
        snippet = "\n".join(chunk_lines)
        chunks.append({
            "text": f"File: {file_path}\nType: general_block\nLines {i+1}-{i+len(chunk_lines)}\n\n{snippet}",
            "file_path": file_path,
            "start_line": i + 1,
            "type": "general_block"
        })
    return chunks
