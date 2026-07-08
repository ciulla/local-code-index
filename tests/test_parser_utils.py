"""
Comprehensive unit tests for parser_utils.py

Covers every public function and every branch thereof:
  - get_language_parser   : supported extensions, aliases, unsupported, ImportError path,
                            AttributeError path, no-match fallthrough.
  - is_valid_test_call    : expression/call_expression nodes with valid test hooks,
                            non-test calls, partial matches, non-expression node types,
                            exception swallowing, byte-slice correctness.
  - get_structural_chunks : real tree-sitter parsing for Python/JS/TS/Java, test-file
                            detection, decorator/annotation/comment lookback, max_lines
                            boundary, unsupported extension, parser failure, empty content,
                            chunks include the expected metadata fields.
  - fallback_line_chunker : basic chunking, overlap step, default args, empty content,
                            single line, trailing empty lines, metadata shape, degenerate
                            max_lines that yields no iterations.

NOTE: We deliberately avoid placing the substrings "test" or "tmp" anywhere inside the
file_path used for general AST tests because ``get_structural_chunks`` flips every chunk
to ``type='test_block'`` whenever ``"test" in file_path.lower()``. Test-file detection
itself is exercised in a dedicated test class below.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from local_code_index.parser_utils import (
    get_language_parser,
    is_valid_test_call,
    get_structural_chunks,
    fallback_line_chunker,
    TARGET_NODE_TYPES,
)


class _FakeNode:
    """Minimal stand-in implementing only the attributes consulted by parser_utils.

    Attributes mirror the real tree_sitter Node API surface we use:
      type, start_byte, end_byte, start_point, end_point, children, prev_sibling, id.
    """

    _id_counter = 0

    def __init__(self, node_type, text, start_byte=0, end_byte=None,
                 start_point=(0, 0), end_point=(0, 0), children=None,
                 prev_sibling=None):
        self.type = node_type
        self.text = text
        self.start_byte = start_byte
        self.end_byte = end_byte if end_byte is not None else len(text.encode("utf-8"))
        self.start_point = start_point
        self.end_point = end_point
        self.children = children or []
        self.prev_sibling = prev_sibling
        _FakeNode._id_counter += 1
        self.id = _FakeNode._id_counter


class TestTargetNodeTypes(unittest.TestCase):
    """Sanity-check the shared constant that drives structural extraction."""

    def test_expected_types_present(self):
        for expected in (
            'class_declaration', 'interface_declaration', 'method_declaration',
            'constructor_declaration', 'enum_declaration', 'class_definition',
            'method_definition', 'function_definition', 'function_declaration',
            'lexical_declaration', 'arrow_function', 'type_alias_declaration',
            'expression_statement', 'call_expression',
        ):
            self.assertIn(expected, TARGET_NODE_TYPES)

    def test_constant_is_a_set(self):
        self.assertIsInstance(TARGET_NODE_TYPES, set)


class TestGetLanguageParser(unittest.TestCase):
    """Test the dynamic runtime importer for official language bindings."""

    def test_javascript_parser(self):
        self.assertIsNotNone(get_language_parser('.js'))

    def test_jsx_alias(self):
        self.assertIsNotNone(get_language_parser('.jsx'))

    def test_typescript_parser(self):
        self.assertIsNotNone(get_language_parser('.ts'))

    def test_tsx_alias(self):
        self.assertIsNotNone(get_language_parser('.tsx'))

    def test_java_parser(self):
        self.assertIsNotNone(get_language_parser('.java'))

    def test_python_parser(self):
        self.assertIsNotNone(get_language_parser('.py'))

    def test_unsupported_extension_returns_none(self):
        self.assertIsNone(get_language_parser('.txt'))

    def test_empty_extension_returns_none(self):
        self.assertIsNone(get_language_parser(''))

    def test_missing_dot_returns_none(self):
        self.assertIsNone(get_language_parser('js'))

    def test_capitalized_extension_is_unsupported_by_importer(self):
        # The importer is case-sensitive: only lowercase extensions match.
        self.assertIsNone(get_language_parser('.PY'))

    def test_import_error_returns_none(self):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            self.assertIsNone(get_language_parser('.py'))

    def test_attribute_error_returns_none(self):
        with patch("builtins.__import__") as mock_import:
            fake_module = MagicMock()
            # Removing the language attribute makes the attribute access raise.
            del fake_module.language
            mock_import.return_value = fake_module
            self.assertIsNone(get_language_parser('.py'))

    def test_returned_object_is_real_parser(self):
        from tree_sitter import Parser
        self.assertIsInstance(get_language_parser('.py'), Parser)


class TestIsValidTestCall(unittest.TestCase):
    """Test the test-framework call filter used by the structural walker."""

    def _make(self, node_type, text):
        """Build a fake node whose byte slice decodes back to ``text``."""
        encoded = text.encode("utf-8")
        return _FakeNode(node_type, text, start_byte=0, end_byte=len(encoded))

    def test_call_expression_describe_is_valid(self):
        self.assertTrue(is_valid_test_call(
            self._make('call_expression', 'describe("suite", () => {})'),
            'describe("suite", () => {})'))

    def test_call_expression_test_is_valid(self):
        self.assertTrue(is_valid_test_call(
            self._make('call_expression', 'test("works", fn)'), 'test("works", fn)'))

    def test_call_expression_it_is_valid(self):
        self.assertTrue(is_valid_test_call(
            self._make('call_expression', 'it("passes", fn)'), 'it("passes", fn)'))

    def test_call_expression_expect_is_valid(self):
        self.assertTrue(is_valid_test_call(
            self._make('call_expression', 'expect(x).toBe(1)'), 'expect(x).toBe(1)'))

    def test_expression_statement_describe_is_valid(self):
        self.assertTrue(is_valid_test_call(
            self._make('expression_statement', 'describe("suite", () => {});'),
            'describe("suite", () => {});'))

    def test_non_test_call_expression_is_invalid(self):
        self.assertFalse(is_valid_test_call(
            self._make('call_expression', 'console.log("hi")'), 'console.log("hi")'))

    def test_non_test_expression_statement_is_invalid(self):
        self.assertFalse(is_valid_test_call(
            self._make('expression_statement', 'myFunction();'), 'myFunction();'))

    def test_partial_match_description_is_invalid(self):
        self.assertFalse(is_valid_test_call(
            self._make('call_expression', 'description("foo")'), 'description("foo")'))

    def test_partial_match_describeNope_is_invalid(self):
        self.assertFalse(is_valid_test_call(
            self._make('call_expression', 'describeNope("foo")'), 'describeNope("foo")'))

    def test_contains_test_without_startswith_is_invalid(self):
        self.assertFalse(is_valid_test_call(
            self._make('call_expression', 'mytest("foo")'), 'mytest("foo")'))

    def test_leading_whitespace_stripped_so_valid(self):
        text = '  describe("x")'
        self.assertTrue(is_valid_test_call(self._make('call_expression', text), text))

    def test_non_expression_node_returns_true(self):
        self.assertTrue(is_valid_test_call(
            self._make('function_definition', 'def f(): pass'), 'def f(): pass'))

    def test_empty_text_expression_returns_false(self):
        self.assertFalse(is_valid_test_call(
            self._make('expression_statement', ''), ''))

    def test_byte_slice_decodes_correctly_with_unicode(self):
        text = 'test("café")'  # multi-byte utf8 inside a valid test hook
        self.assertTrue(is_valid_test_call(self._make('call_expression', text), text))

    def test_exception_returns_false(self):
        """If the byte slice raises, the function must swallow it and return False."""

        class _ExplodingNode:
            type = 'expression_statement'
            start_byte = 0
            end_byte = 10

        with patch("local_code_index.parser_utils.bytes", side_effect=RuntimeError("boom")):
            result = is_valid_test_call(_ExplodingNode(), "describe(")
        self.assertFalse(result)


class TestGetStructuralChunksPython(unittest.TestCase):
    """Real tree-sitter parsing of Python source."""

    def test_python_function_and_class_chunking(self):
        # The subtree is marked visited when a top-level node is chunked, so a class
        # will collapse into a single chunk rather than emitting one per method.
        code = (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "class Calculator:\n"
            "    def multiply(self, a, b):\n"
            "        return a * b\n"
            "    def divide(self, a, b):\n"
            "        return a / b\n"
        )
        chunks = get_structural_chunks("/work/calc.py", code)

        self.assertEqual(len(chunks), 2)
        types = [c['type'] for c in chunks]
        self.assertEqual(types.count('function_definition'), 1)
        self.assertEqual(types.count('class_definition'), 1)

        first = chunks[0]
        self.assertIn('def add', first['text'])
        self.assertEqual(first['file_path'], "/work/calc.py")
        self.assertEqual(first['start_line'], 1)
        for c in chunks:
            self.assertIn('File:', c['text'])
            self.assertIn('Type:', c['text'])
            self.assertIn('Lines', c['text'])

    def test_python_decorator_lookback(self):
        code = (
            "@decorator\n"
            "def decorated():\n"
            "    return 1\n"
        )
        chunks = get_structural_chunks("/work/dec.py", code)
        self.assertEqual(len(chunks), 1)
        first = chunks[0]
        self.assertIn('@decorator', first['text'])
        self.assertEqual(first['start_line'], 1)
        self.assertEqual(first['type'], 'function_definition')

    def test_python_function_too_long_returns_no_chunk(self):
        long_body = "def big():\n" + "\n".join(f"    x = {i}" for i in range(200))
        self.assertEqual(len(get_structural_chunks("/work/big.py", long_body)), 0)

    def test_python_function_within_max_lines_chunked(self):
        body = "def small():\n" + "\n".join(f"    x = {i}" for i in range(20))
        chunks = get_structural_chunks("/work/small.py", body)
        self.assertEqual(len(chunks), 1)
        self.assertIn('def small', chunks[0]['text'])

    def test_python_max_lines_custom(self):
        code = (
            "def f():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    d = 4\n"
            "    e = 5\n"
            "    g = 6\n"
        )
        self.assertEqual(len(get_structural_chunks("/work/f.py", code, max_lines=80)), 1)
        self.assertEqual(len(get_structural_chunks("/work/f.py", code, max_lines=3)), 0)


class TestGetStructuralChunksJavaScript(unittest.TestCase):
    """Real tree-sitter parsing of JS/TS source (paths avoid the 'test' keyword)."""

    def test_javascript_function_and_arrow_chunking(self):
        code = (
            "function add(a, b) {\n"
            "    return a + b;\n"
            "}\n"
            "\n"
            "const multiply = (a, b) => {\n"
            "    return a * b;\n"
            "};\n"
        )
        chunks = get_structural_chunks("/work/calc.js", code)
        self.assertEqual(len(chunks), 2)
        types = [c['type'] for c in chunks]
        self.assertIn('function_declaration', types)
        # The arrow is wrapped in a lexical_declaration that gets visited first as a
        # TARGET_NODE_TYPE and its subtree (the arrow_function) is marked visited.
        self.assertIn('lexical_declaration', types)

    def test_javascript_test_calls_recognised_precedence(self):
        # The describe() block, being an expression_statement that is_valid_test_call
        # accepts, is chunked at the outer level; the inner test()/expect() calls live
        # inside that subtree and are therefore marked visited so they don't surface.
        code = (
            "describe('suite', () => {\n"
            "    it('works', () => {\n"
            "        expect(1).toBe(1);\n"
            "    });\n"
            "});\n"
        )
        chunks = get_structural_chunks("/work/runner.spec.js", code)
        self.assertEqual(len(chunks), 1)
        # Filename contains neither 'test' nor an expression node type that maps to
        # 'test_block', so the chunk type isn't forcibly overridden to test_block
        # here. The expression_statement/call_expression node produces a "test_block"
        # because of the in-predicate check in get_structural_chunks.
        self.assertEqual(chunks[0]['type'], 'test_block')

    def test_javascript_non_test_calls_not_chunked_at_top(self):
        code = (
            "function helper() {\n"
            "    return 1;\n"
            "}\n"
            "console.log(helper());\n"
        )
        chunks = get_structural_chunks("/work/helpers.js", code)
        types = [c['type'] for c in chunks]
        self.assertEqual(types, ['function_declaration'])

    def test_typescript_interface_and_type_alias(self):
        code = (
            "interface User {\n"
            "    name: string;\n"
            "    age: number;\n"
            "}\n"
            "\n"
            "type ID = string | number;\n"
        )
        chunks = get_structural_chunks("/work/types.ts", code)
        types = [c['type'] for c in chunks]
        self.assertIn('interface_declaration', types)
        self.assertIn('type_alias_declaration', types)


class TestGetStructuralChunksJava(unittest.TestCase):
    """Real tree-sitter parsing of Java source."""

    def test_java_class_collapses_to_single_chunk(self):
        code = (
            "public class Calculator {\n"
            "    public int add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "    private int multiply(int a, int b) {\n"
            "        return a * b;\n"
            "    }\n"
            "}\n"
        )
        chunks = get_structural_chunks("/work/Calculator.java", code)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['type'], 'class_declaration')
        self.assertIn('Calculator', chunks[0]['text'])
        self.assertIn('add', chunks[0]['text'])
        self.assertIn('multiply', chunks[0]['text'])

    def test_java_annotation_lookback(self):
        code = (
            "@Override\n"
            "public String toString() {\n"
            "    return \"x\";\n"
            "}\n"
        )
        chunks = get_structural_chunks("/work/A.java", code)
        self.assertEqual(len(chunks), 1)
        self.assertIn('@Override', chunks[0]['text'])
        self.assertEqual(chunks[0]['start_line'], 1)
        self.assertEqual(chunks[0]['type'], 'method_declaration')


class TestStructuralChunksTestFileDetection(unittest.TestCase):
    """The ``"test" in file_path.lower()`` branch forces ``type='test_block'``."""

    def test_test_in_path_flips_every_chunk_to_test_block(self):
        code = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
        )
        chunks = get_structural_chunks("/work/CalculatorSpec_Test.py", code)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c['type'], 'test_block')

    def test_uppercase_TEST_in_path_also_flips(self):
        code = "function helper() { return 1; }\n"
        chunks = get_structural_chunks("/work/TEST_HELP.js", code)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c['type'], 'test_block')


class TestGetStructuralChunksEdges(unittest.TestCase):
    """Edge cases that don't depend on a specific language parser."""

    def test_unsupported_extension_returns_empty(self):
        self.assertEqual(
            get_structural_chunks("/work/file.txt", "hello world"), []
        )

    def test_extension_uppercased_still_works_due_to_lower_casing(self):
        # get_structural_chunks lowercases os.path.splitext's extension before
        # dispatching, so .PY ends up as .py and is parsed correctly.
        chunks = get_structural_chunks("/work/file.PY", "def f():\n    pass")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['type'], 'function_definition')

    def test_empty_content_returns_empty(self):
        self.assertEqual(get_structural_chunks("/work/empty.py", ""), [])

    def test_parser_failure_returns_empty(self):
        with patch("local_code_index.parser_utils.get_language_parser", return_value=None):
            self.assertEqual(
                get_structural_chunks("/work/x.py", "def f(): pass"), []
            )

    def test_parse_exception_swallowed(self):
        fake_parser = MagicMock()
        fake_parser.parse.side_effect = RuntimeError("boom")
        with patch("local_code_index.parser_utils.get_language_parser", return_value=fake_parser):
            self.assertEqual(
                get_structural_chunks("/work/x.py", "def f(): pass"), []
            )


class TestFallbackLineChunker(unittest.TestCase):
    """Test the line-based chunker used when AST parsing yields nothing."""

    def test_single_chunk_when_content_under_max_lines(self):
        content = "\n".join(f"Line {i}" for i in range(1, 21))  # 20 lines
        chunks = fallback_line_chunker("test.txt", content, max_lines=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0]['text'],
            "File: test.txt\nType: general_block\nLines 1-20\n\n" + content,
        )
        self.assertEqual(chunks[0]['file_path'], "test.txt")
        self.assertEqual(chunks[0]['start_line'], 1)
        self.assertEqual(chunks[0]['type'], "general_block")

    def test_two_chunks_when_content_exceeds_max_lines(self):
        content = "\n".join(f"Line {i}" for i in range(1, 101))  # 100 lines
        chunks = fallback_line_chunker("f.txt", content, max_lines=80)
        # step = max_lines - 15 = 65, so chunks at offsets 0 and 65.
        self.assertEqual(len(chunks), 2)
        self.assertIn("Lines 1-80", chunks[0]['text'])
        self.assertIn("Lines 66-100", chunks[1]['text'])

    def test_large_content_single_chunk(self):
        # No defaults are defined on the function, but realistic single-chunk case.
        content = "\n".join(f"Line {i}" for i in range(1, 50))  # 49 lines
        chunks = fallback_line_chunker("f.txt", content, max_lines=80)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Lines 1-49", chunks[0]['text'])

    def test_empty_content_returns_empty(self):
        self.assertEqual(fallback_line_chunker("f.txt", "", max_lines=80), [])

    def test_single_line(self):
        chunks = fallback_line_chunker("f.txt", "Only line", max_lines=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['start_line'], 1)
        self.assertIn("Only line", chunks[0]['text'])

    def test_content_with_trailing_newline(self):
        content = "a\nb\n"  # splitlines -> ['a', 'b']
        chunks = fallback_line_chunker("f.txt", content, max_lines=80)
        self.assertEqual(len(chunks), 1)
        self.assertIn("a\nb", chunks[0]['text'])

    def test_chunk_metadata_shape(self):
        content = "a\nb\nc"
        chunks = fallback_line_chunker("path/x.py", content, max_lines=80)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(set(c.keys()), {"text", "file_path", "start_line", "type"})
        self.assertEqual(c['file_path'], "path/x.py")
        self.assertEqual(c['type'], "general_block")

    def test_small_max_lines_step_negative_yields_no_chunks(self):
        # step = max_lines - 15 = -13 -> range(0, n, -13) yields no iterations -> []
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        self.assertEqual(fallback_line_chunker("test.txt", content, max_lines=2), [])


if __name__ == '__main__':
    unittest.main()
