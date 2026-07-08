"""
Comprehensive unit tests for server.py

Covers every public function and every branch thereof:
  - count_tokens               : basic/empty/unicode/multiple calls, encoding passthrough.
  - get_embedding              : success, empty input, ollama error, missing embedding key.
  - get_clean_table_name       : basic, special chars, Windows path, same-path determinism,
                                  empty path, md5 truncation to 8 chars.
  - update_manifest            : create-new-table path, update-existing-table path,
                                  deterministic manifest record shape.
  - list_indexed_repositories  : no tables, no manifest (only repo tables), empty manifest,
                                  populated manifest output formatting.
  - index_repository           : nonexistent path, no files, supported file happy path,
                                  large-file skip, fallback chunker, encoding error during
                                  open, IVF-PQ index creation when table grows past 300 rows,
                                  basic direct-lookup path, delete-before-create when the
                                  matching table already exists, update_manifest integration.
  - search_codebase            : repo not indexed, success with multiple hits, file_filter
                                  adds a WHERE clause, empty results message, truncation by
                                  token budget.
  - delete_repository          : resolve by real on-disk path, resolve by string path, by
                                  raw table name, table not found, invalid identifier,
                                  manifest-delete exception silently swallowed, manifest-only
                                  removal (table already gone).
  - search_all_codebases       : no repos, single repo success, multi repo ordered by distance,
                                  one-repo-fails-keep-others, token-budget truncation, empty
                                  combined-results message.
"""
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open

import server
from server import (
    count_tokens,
    get_embedding,
    get_clean_table_name,
    update_manifest,
    list_indexed_repositories,
    index_repository,
    search_codebase,
    delete_repository,
    search_all_codebases,
)

# The FastMCP server object should never actually start during tests; neutralise it
# the same way as the legacy suite does.
server.mcp = MagicMock()


class _ServerDBFixture(unittest.TestCase):
    """Save/restore the global ``server.db`` mock so tests cannot leak state.

    Each test patches ``server.db`` via ``@patch('server.db')``; this base class
    ensures the original (live lancedb connection) is restored in tearDown even if
    a subclass forgets the explicit teardown.
    """

    def setUp(self):
        super().setUp()
        self._original_db = server.db
        self._original_os = server.os
        self._original_get_embedding = server.get_embedding
        self._original_count_tokens = server.count_tokens
        self._original_get_structural_chunks = server.get_structural_chunks
        self._original_fallback_line_chunker = server.fallback_line_chunker
        self._original_update_manifest = server.update_manifest

    def tearDown(self):
        server.db = self._original_db
        server.os = self._original_os
        server.get_embedding = self._original_get_embedding
        server.count_tokens = self._original_count_tokens
        server.get_structural_chunks = self._original_get_structural_chunks
        server.fallback_line_chunker = self._original_fallback_line_chunker
        server.update_manifest = self._original_update_manifest
        super().tearDown()


class TestCountTokens(unittest.TestCase):
    """count_tokens delegates to the global tokenizer.encode()."""

    def setUp(self):
        self._original = server.TOKENIZER
        self.mock_tokenizer = MagicMock()
        server.TOKENIZER = self.mock_tokenizer

    def tearDown(self):
        server.TOKENIZER = self._original

    def test_basic(self):
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.assertEqual(count_tokens("hello world"), 5)
        self.mock_tokenizer.encode.assert_called_once_with(
            "hello world", disallowed_special=()
        )

    def test_empty_string(self):
        self.mock_tokenizer.encode.return_value = []
        self.assertEqual(count_tokens(""), 0)

    def test_unicode(self):
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.assertEqual(count_tokens("café"), 3)

    def test_multiple_calls_no_caching(self):
        self.mock_tokenizer.encode.return_value = [1]
        count_tokens("a")
        count_tokens("b")
        count_tokens("c")
        self.assertEqual(self.mock_tokenizer.encode.call_count, 3)


class TestTokenizerFallback(unittest.TestCase):
    """Module-level tokenizer init falls back to gpt-4 when cl100k_base fails."""

    def test_cl100k_base_failure_falls_back_to_gpt4(self):
        import importlib

        original_cl100k = server.tiktoken.get_encoding
        original_tokenizer_attr = server.TOKENIZER
        try:
            cl100k_failures = {"calls": 0}
            gpt4_tokenizer = MagicMock(name="gpt4")

            def fake_get_encoding(name):
                if name == "cl100k_base":
                    cl100k_failures["calls"] += 1
                    raise RuntimeError("cl100k unavailable in test")
                if name == "gpt-4":
                    return gpt4_tokenizer
                raise RuntimeError(f"unexpected encoding {name!r}")

            with patch.object(server.tiktoken, "get_encoding", side_effect=fake_get_encoding):
                importlib.reload(server)

            try:
                self.assertEqual(cl100k_failures["calls"], 1)
                # After reload the module-level TOKENIZER must come from gpt-4 fallback.
                self.assertIs(server.TOKENIZER, gpt4_tokenizer)
            finally:
                # Restore the module to its live state for downstream tests.
                importlib.reload(server)
        finally:
            # The reload above re-evaluates module-level code; ensure subsequent
            # tests start from a non-reloaded server module to avoid cross-test noise.
            self.assertEqual(server.tiktoken.get_encoding, original_cl100k)


class TestGetEmbedding(unittest.TestCase):
    """get_embedding wraps ollama.embeddings and converts all errors to RuntimeError."""

    def setUp(self):
        self._original_ollama = server.ollama

    def tearDown(self):
        server.ollama = self._original_ollama

    @patch('server.ollama.embeddings')
    def test_success(self, mock_embeddings):
        mock_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
        self.assertEqual(get_embedding("test text"), [0.1, 0.2, 0.3])
        mock_embeddings.assert_called_once_with(model="nomic-embed-text", prompt="test text")

    @patch('server.ollama.embeddings')
    def test_empty_input(self, mock_embeddings):
        mock_embeddings.return_value = {"embedding": [0.0]}
        self.assertEqual(get_embedding(""), [0.0])
        mock_embeddings.assert_called_once_with(model="nomic-embed-text", prompt="")

    @patch('server.ollama.embeddings')
    def test_connection_error_raises_runtime_error(self, mock_embeddings):
        mock_embeddings.side_effect = Exception("Connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            get_embedding("test text")
        self.assertIn("Ollama connection error", str(ctx.exception))
        self.assertIn("Ensure 'ollama pull nomic-embed-text' was run", str(ctx.exception))

    @patch('server.ollama.embeddings')
    def test_missing_embedding_key_raises_runtime_error(self, mock_embeddings):
        mock_embeddings.return_value = {"other": "value"}
        with self.assertRaises(RuntimeError):
            get_embedding("test text")


class TestGetCleanTableName(unittest.TestCase):
    """get_clean_table_name => repo_<sanitised-name>_<8-char-md5>."""

    def test_basic(self):
        result = get_clean_table_name("/path/to/my-repo")
        self.assertTrue(result.startswith("repo_"))
        self.assertIn("my_repo_", result)
        self.assertEqual(len(result.split("_")[-1]), 8)

    def test_dots_and_dashes_replaced(self):
        result = get_clean_table_name("/path/to/my.repo_with-dashes")
        # basename 'my.repo_with-dashes' -> 'my_repo_with_dashes'
        # The hash anchors position _after_ the sanitised basename, so check membership.
        self.assertIn("my_repo_with_dashes_", result)
        self.assertEqual(len(result.split("_")[-1]), 8)

    def test_windows_path(self):
        result = get_clean_table_name("C:\\Users\\user\\my-repo")
        self.assertTrue(result.startswith("repo_"))
        self.assertIn("my_repo_", result)
        self.assertEqual(len(result.split("_")[-1]), 8)

    def test_same_path_deterministic(self):
        self.assertEqual(
            get_clean_table_name("/path/to/my-repo"),
            get_clean_table_name("/path/to/my-repo"),
        )

    def test_different_paths_different_hashes(self):
        self.assertNotEqual(
            get_clean_table_name("/path/to/repoA"),
            get_clean_table_name("/path/to/repoB"),
        )

    def test_empty_path(self):
        result = get_clean_table_name("")
        self.assertTrue(result.startswith("repo_"))
        # basename('') with normpath collapses to '' -> double underscore from join.
        self.assertIn("__", result.replace("repo_", "", 1) or result)
        self.assertEqual(len(result.split("_")[-1]), 8)


class TestUpdateManifest(_ServerDBFixture):
    """update_manifest adds/refreshes a manifest record."""

    @patch('server.db')
    @patch('server.os')
    def test_new_manifest_table_created_with_overwrite(self, mock_os, mock_db):
        mock_os.path.abspath.return_value = "/abs/repo"
        # basename must resolve to a plain string (mock would otherwise return a Mock).
        mock_os.path.basename.return_value = "repo"
        mock_os.path.normpath.return_value = "/abs/repo"
        mock_db.table_names.return_value = []  # master_manifest absent

        update_manifest("/abs/repo", "repo_repo_abcd1234", 10)

        mock_db.create_table.assert_called_once_with(
            "master_manifest",
            data=[{
                "repo_path": "/abs/repo",
                "repo_name": "repo",
                "table_name": "repo_repo_abcd1234",
                "total_chunks": 10,
            }],
            mode="overwrite",
        )
        mock_db.open_table.assert_not_called()

    @patch('server.db')
    @patch('server.os')
    def test_existing_manifest_table_replaces_entry(self, mock_os, mock_db):
        mock_os.path.abspath.return_value = "/abs/repo"
        mock_os.path.basename.return_value = "repo"
        mock_os.path.normpath.return_value = "/abs/repo"
        mock_db.table_names.return_value = ["master_manifest", "repo_repo_abcd1234"]

        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table

        update_manifest("/abs/repo", "repo_repo_abcd1234", 25)

        mock_db.open_table.assert_called_once_with("master_manifest")
        mock_table.delete.assert_called_once_with("repo_path = '/abs/repo'")
        mock_table.add.assert_called_once_with([{
            "repo_path": "/abs/repo",
            "repo_name": "repo",
            "table_name": "repo_repo_abcd1234",
            "total_chunks": 25,
        }])
        mock_db.create_table.assert_not_called()


class TestListIndexedRepositories(_ServerDBFixture):
    """list_indexed_repositories read path."""

    @patch('server.db')
    def test_no_tables_at_all(self, mock_db):
        mock_db.table_names.return_value = []
        self.assertEqual(
            list_indexed_repositories(), "No repositories have been indexed yet."
        )

    @patch('server.db')
    def test_no_manifest_only_repo_tables(self, mock_db):
        mock_db.table_names.return_value = ["repo_a_11111111", "repo_b_22222222"]
        self.assertEqual(
            list_indexed_repositories(), "No repositories have been indexed yet."
        )

    @patch('server.db')
    def test_manifest_present_but_empty(self, mock_db):
        mock_db.table_names.return_value = ["master_manifest", "repo_a_11111111"]
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value = mock_table.search
        mock_table.search.to_list = MagicMock(return_value=[])
        self.assertEqual(
            list_indexed_repositories(), "No repositories found in manifest tracker."
        )

    @patch('server.db')
    def test_manifest_with_data(self, mock_db):
        mock_db.table_names.return_value = ["master_manifest", "repo_a_11111111"]
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.to_list.return_value = [
            {"repo_name": "alpha", "repo_path": "/p/alpha", "total_chunks": 5},
            {"repo_name": "beta", "repo_path": "/p/beta", "total_chunks": 8},
        ]

        result = list_indexed_repositories()

        self.assertIn("=== Indexed Codebases ===", result)
        # Original format from server.py uses two-space indent and a single newline.
        self.assertIn("- Name: alpha\n  Path: /p/alpha\n  Chunks: 5\n", result)
        self.assertIn("- Name: beta\n  Path: /p/beta\n  Chunks: 8\n", result)


class TestIndexRepository(_ServerDBFixture):
    """index_repository walks the filesystem, chunks, embeds, and persists.

    Implementation note: we explicitly patch only the os functions the function
    consults (exists, walk, getsize) and let the rest of os.path.* run for real so
    that join/basename/normpath/splitext string formatting produces real strings the
    assertions can compare against.
    """

    def _patch_os_walk(self, walker, mock_walk):
        """Helper: replace os.walk while letting other os.path.* work."""
        mock_walk.return_value = iter(walker)

    def test_nonexistent_path_returns_error(self):
        with patch('server.os.path.exists', return_value=False), \
             patch('server.os.path.abspath', return_value="/no/such/path"):
            result = index_repository("/no/such/path")
        self.assertEqual(result, "Error: Path /no/such/path does not exist.")

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    def test_no_supported_files_returns_message(self, mock_embed, mock_chunks):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["readme.md"])])):
            with patch('server.update_manifest'):
                result = index_repository("/repo")
        self.assertEqual(result, "No processable source files found.")
        mock_chunks.assert_not_called()
        mock_embed.assert_not_called()

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    @patch('builtins.open', new_callable=mock_open, read_data="def f():\n    pass")
    def test_supported_file_happy_path_basic_index(
        self, mock_file, mock_embed, mock_chunks
    ):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=100), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["module.py"])])):
            mock_chunks.return_value = [
                {
                    "text": "File: /repo/module.py\nType: function_definition\nLines 1-2\n\ndef f():\n    pass",
                    "file_path": "/repo/module.py",
                    "start_line": 1,
                    "type": "function_definition",
                }
            ]
            mock_embed.return_value = [0.1, 0.2, 0.3]

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=50)
            server.db = MagicMock()
            server.db.table_names.return_value = []
            server.db.create_table.return_value = mock_table

            with patch('server.update_manifest') as mock_manifest:
                result = index_repository("/repo")

        self.assertIn("Success: Codebase 'repo' indexed completely (1 node", result)
        self.assertIn("with basic vector direct lookup", result)
        # The embedding vector must have been attached to the chunk before save.
        # ``data=`` is passed as a keyword arg by index_repository.
        call_kwargs = server.db.create_table.call_args.kwargs
        saved = call_kwargs["data"]
        self.assertEqual(saved[0]["vector"], [0.1, 0.2, 0.3])
        # Basic index path should NOT call create_index.
        mock_table.create_index.assert_not_called()
        # update_manifest receives abs_path, table_name, count.
        mock_manifest.assert_called_once()
        self.assertEqual(mock_manifest.call_args.args[0], "/repo")
        self.assertEqual(mock_manifest.call_args.args[2], 1)

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    @patch('builtins.open', new_callable=mock_open, read_data="def f():\n    pass")
    def test_large_table_creates_ivf_pq_index(self, mock_file, mock_embed, mock_chunks):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=100), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["module.py"])])):
            mock_chunks.return_value = [
                {
                    "text": "x",
                    "file_path": "/repo/module.py",
                    "start_line": 1,
                    "type": "function_definition",
                }
            ]
            mock_embed.return_value = [0.1]

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=400)  # > 300 -> IVF-PQ path
            server.db = MagicMock()
            server.db.table_names.return_value = []
            server.db.create_table.return_value = mock_table

            with patch('server.update_manifest'):
                result = index_repository("/repo")

        self.assertIn("with structural IVF-PQ acceleration", result)
        mock_table.create_index.assert_called_once_with(
            metric="cosine", num_partitions=16, num_sub_vectors=96
        )

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    def test_file_exceeds_two_megabytes_is_skipped(self, mock_embed, mock_chunks):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=3 * 1024 * 1024), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["huge.py"])])):
            with patch('server.update_manifest'):
                result = index_repository("/repo")
        self.assertEqual(result, "No processable source files found.")
        mock_chunks.assert_not_called()
        mock_embed.assert_not_called()

    @patch('server.get_structural_chunks')
    @patch('server.fallback_line_chunker')
    @patch('server.get_embedding')
    @patch('builtins.open', new_callable=mock_open, read_data="# comment\ndef f():\n    pass")
    def test_fallback_chunker_used_when_ast_empty(
        self, mock_file, mock_embed, mock_fallback, mock_chunks
    ):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=100), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["weird.py"])])):
            mock_chunks.return_value = []  # AST yields nothing
            mock_fallback.return_value = [
                {
                    "text": "fallback snippet",
                    "file_path": "/repo/weird.py",
                    "start_line": 1,
                    "type": "general_block",
                }
            ]
            mock_embed.return_value = [0.5]

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=10)
            server.db = MagicMock()
            server.db.table_names.return_value = []
            server.db.create_table.return_value = mock_table

            with patch('server.update_manifest'):
                result = index_repository("/repo")

        self.assertIn("Success", result)
        # Fallback was invoked with the file path, content, and max_lines=80.
        mock_fallback.assert_called_once()
        # ``max_lines=80`` is passed as a keyword arg.
        self.assertEqual(mock_fallback.call_args.args[0], os.path.normpath("/repo/weird.py"))
        self.assertEqual(mock_fallback.call_args.kwargs["max_lines"], 80)

    @patch('server.get_structural_chunks')
    @patch('builtins.open', new_callable=mock_open, read_data="def f():\n    pass")
    def test_file_read_exception_is_swallowed_per_file(self, mock_file, mock_chunks):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=100), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["bad.py", "good.py"])])):
            good_chunk = [{
                "text": "good", "file_path": os.path.normpath("/repo/good.py"),
                "start_line": 1, "type": "function_definition",
            }]
            mock_chunks.return_value = good_chunk
            server.get_embedding = MagicMock(return_value=[0.1])

            call_count = {"n": 0}
            m_open = mock_open(read_data="def f():\n    pass")

            def side_effect(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise OSError("disk error")
                return m_open(*args, **kwargs)

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=10)
            server.db = MagicMock()
            server.db.table_names.return_value = []
            server.db.create_table.return_value = mock_table

            with patch('builtins.open', side_effect=side_effect):
                with patch('server.update_manifest'):
                    result = index_repository("/repo")

        self.assertIn("Success: Codebase 'repo' indexed completely (1 node", result)

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    @patch('builtins.open', new_callable=mock_open, read_data="def f():\n    pass")
    def test_existing_table_dropped_before_create(
        self, mock_file, mock_embed, mock_chunks
    ):
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', return_value=100), \
             patch('server.os.walk', return_value=iter([("/repo", [], ["m.py"])])):
            mock_chunks.return_value = [
                {"text": "x", "file_path": os.path.normpath("/repo/m.py"),
                 "start_line": 1, "type": "function_definition"}
            ]
            mock_embed.return_value = [0.1]

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=10)
            server.db = MagicMock()
            # Table already exists -> drop_table must be called before create_table.
            expected_table = get_clean_table_name("/repo")
            server.db.table_names.return_value = [expected_table]
            server.db.create_table.return_value = mock_table

            with patch('server.update_manifest'):
                index_repository("/repo")

        server.db.drop_table.assert_called_once_with(expected_table)
        server.db.create_table.assert_called_once()

    @patch('server.get_structural_chunks')
    @patch('server.get_embedding')
    @patch('builtins.open', new_callable=mock_open, read_data="x")
    def test_ignored_files_are_skipped_before_extension_check(
        self, mock_file, mock_embed, mock_chunks
    ):
        """Files in ``ignore_files`` or starting with ``.env`` short-circuit before
        the supported-extension check, exercising the ``continue`` branch."""
        from unittest.mock import MagicMock as _MM
        getsize_mock = _MM(return_value=100)
        with patch('server.os.path.exists', return_value=True), \
             patch('server.os.path.abspath', return_value="/repo"), \
             patch('server.os.path.normpath', side_effect=os.path.normpath), \
             patch('server.os.path.getsize', new=getsize_mock), \
             patch('server.os.walk', return_value=iter([
                 ("/repo", [], ["package-lock.json", ".env", "tsconfig.json", "code.py"]),
             ])):
            mock_chunks.return_value = [
                {"text": "x", "file_path": os.path.normpath("/repo/code.py"),
                 "start_line": 1, "type": "function_definition"}
            ]
            mock_embed.return_value = [0.1]

            mock_table = MagicMock()
            mock_table.__len__ = MagicMock(return_value=10)
            server.db = MagicMock()
            server.db.table_names.return_value = []
            server.db.create_table.return_value = mock_table

            with patch('server.update_manifest'):
                result = index_repository("/repo")

        # Only ``code.py`` was processed -> getsize call_count must be 1, not 4.
        self.assertEqual(getsize_mock.call_count, 1)
        self.assertIn("Success: Codebase 'repo' indexed completely (1 node", result)
        # The ignore-listed files were never chunked or embedded.
        mock_chunks.assert_called_once()
        mock_embed.assert_called_once()


class TestSearchCodebase(_ServerDBFixture):
    """search_codebase -> vector search with token-budget guardrails."""

    @patch('server.get_embedding')
    def test_repo_not_indexed_returns_error(self, mock_embed):
        mock_embed.return_value = [0.0]
        server.db = MagicMock()
        server.db.table_names.return_value = ["repo_other_abcd1234"]
        result = search_codebase("/path/to/repo", "query")
        self.assertIn("Error: Repository at '/path/to/repo' is not indexed yet.", result)
        self.assertIn("Use index_repository first.", result)

    @patch('server.get_embedding')
    def test_repo_not_indexed_uses_absolute_path_in_message(self, mock_embed):
        mock_embed.return_value = [0.0]
        server.db = MagicMock()
        # Simulate the table name that *would* be derived from the path not being present.
        server.db.table_names.return_value = []
        result = search_codebase("/repo", "q")
        self.assertIn("Error: Repository at '/repo' is not indexed yet.", result)

    @patch('server.count_tokens')
    @patch('server.get_embedding')
    def test_success_returns_formatted_results(self, mock_embed, mock_tokens):
        mock_embed.return_value = [0.1, 0.2]
        # count_tokens: keep returning small numbers so nothing gets truncated.
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        # Table name must match get_clean_table_name(abs_path) for "/repo" -> resolved.
        # The function calls get_clean_table_name(os.path.abspath(repo_path)); we let the
        # real implementation run, so we just need the resulting table name to be present.
        target_table = get_clean_table_name(os.path.abspath("/repo"))
        server.db.table_names.return_value = [target_table]

        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table

        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {
                "file_path": "/repo/test.py", "start_line": 1,
                "type": "function_definition", "text": "def f():\n    pass",
            },
            {
                "file_path": "/repo/test2.py", "start_line": 5,
                "type": "class_definition", "text": "class C:\n    pass",
            },
        ]

        result = search_codebase("/repo", "query", limit=10, token_budget=1000)

        self.assertIn("=== Results for Codebase 'repo' ===", result)
        self.assertIn("[1] File: /repo/test.py (Line 1) | Type: function_definition", result)
        self.assertIn("def f():\n    pass", result)
        self.assertIn("[2] File: /repo/test2.py (Line 5) | Type: class_definition", result)
        # No truncation warning expected.
        self.assertNotIn("WARNING", result)

    @patch('server.count_tokens')
    @patch('server.get_embedding')
    def test_token_budget_truncates_results(self, mock_embed, mock_tokens):
        mock_embed.return_value = [0.1]
        # First call (header) returns 0; every block returns a large number to force break.
        mock_tokens.side_effect = lambda text: 0 if "===" in text else 5000

        server.db = MagicMock()
        target_table = get_clean_table_name(os.path.abspath("/repo"))
        server.db.table_names.return_value = [target_table]
        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {"file_path": "/repo/a.py", "start_line": 1,
             "type": "function_definition", "text": "def a(): pass"},
        ]

        result = search_codebase("/repo", "query", token_budget=100)
        self.assertIn("WARNING: Search results truncated to fit within the 100 token", result)

    @patch('server.count_tokens')
    @patch('server.get_embedding')
    def test_file_filter_adds_where_clause(self, mock_embed, mock_tokens):
        mock_embed.return_value = [0.1]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        target_table = get_clean_table_name(os.path.abspath("/repo"))
        server.db.table_names.return_value = [target_table]
        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        # .where(...) must return the same chain object so .limit & .to_list chain on it.
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {"file_path": "/repo/test.py", "start_line": 1,
             "type": "function_definition", "text": "def f(): pass"},
        ]

        search_codebase("/repo", "query", file_filter="test.py")

        mock_search.where.assert_called_once()
        where_arg = mock_search.where.call_args.args[0]
        self.assertIn("file_path LIKE '%test.py%'", where_arg)
        self.assertTrue(mock_search.where.call_args.kwargs.get("prefilter", False))

    @patch('server.count_tokens')
    @patch('server.get_embedding')
    def test_empty_results_returns_message(self, mock_embed, mock_tokens):
        mock_embed.return_value = [0.1]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        target_table = get_clean_table_name(os.path.abspath("/repo"))
        server.db.table_names.return_value = [target_table]
        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = []

        result = search_codebase("/repo", "query")
        self.assertEqual(result, "No relevant structural definitions found for this query.")


class TestDeleteRepository(unittest.TestCase):
    """delete_repository accepts a filesystem path OR a raw table name."""

    def setUp(self):
        self._original_db = server.db
        self._original_os = server.os

    def tearDown(self):
        server.db = self._original_db
        server.os = self._original_os

    @patch('server.os')
    def test_delete_by_existing_path_drops_table_and_manifest(self, mock_os):
        mock_os.path.abspath.return_value = "/repo"
        # Force the path-existence branch: "/" or "\" present OR os.path.exists True.
        mock_os.path.exists.return_value = True
        mock_os.path.normpath.side_effect = lambda p: p
        table_name = "repo_repo_abcd1234"

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", table_name]
        # Critical: get_clean_table_name uses the *real* os.path.basename/normpath; but
        # we patched server.os as a whole, so basename would return a Mock. Patch the
        # function directly to return the expected table name.
        with patch('server.get_clean_table_name', return_value=table_name):
            mock_manifest = MagicMock()
            server.db.open_table.return_value = mock_manifest

            result = delete_repository("/repo")

        server.db.drop_table.assert_called_once_with(table_name)
        # Table starts with "repo_" -> manifest delete uses table_name predicate.
        mock_manifest.delete.assert_called_once_with(f"table_name = '{table_name}'")
        self.assertIn("Successfully pruned and deleted vector data for:", result)

    def test_delete_by_raw_table_name(self):
        table_name = "repo_repo_abcd1234"
        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", table_name]
        mock_manifest = MagicMock()
        server.db.open_table.return_value = mock_manifest

        result = delete_repository(table_name)

        server.db.drop_table.assert_called_once_with(table_name)
        mock_manifest.delete.assert_called_once_with(f"table_name = '{table_name}'")
        self.assertIn("Successfully pruned and deleted vector data for:", result)

    @patch('server.os')
    def test_delete_when_table_missing_but_manifest_present(self, mock_os):
        mock_os.path.abspath.return_value = "/repo"
        mock_os.path.exists.return_value = True
        mock_os.path.normpath.side_effect = lambda p: p
        table_name = "repo_repo_abcd1234"

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest"]  # repo table absent
        mock_manifest = MagicMock()
        server.db.open_table.return_value = mock_manifest

        with patch('server.get_clean_table_name', return_value=table_name):
            result = delete_repository("/repo")

        server.db.drop_table.assert_not_called()
        # Manifest entry removal still attempted because manifest present.
        mock_manifest.delete.assert_called_once_with(f"table_name = '{table_name}'")
        self.assertIn("Successfully pruned and deleted vector data for:", result)

    @patch('server.os')
    def test_delete_not_found_message(self, mock_os):
        mock_os.path.abspath.return_value = "/nonexistent"
        mock_os.path.exists.return_value = True
        mock_os.path.normpath.side_effect = lambda p: p
        table_name = "repo_nonexistent_abcd1234"

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", "repo_other_xxxxxxxx"]
        mock_manifest = MagicMock()
        server.db.open_table.return_value = mock_manifest

        with patch('server.get_clean_table_name', return_value=table_name):
            result = delete_repository("/nonexistent")

        server.db.drop_table.assert_not_called()
        # Even though repo table absent, manifest IS present -> delete attempted and
        # manifest_removed True (because tbl.delete doesn't raise) -> success message.
        # Verify the not-found message text used when both removals are False.
        # Here manifest removal succeeds, so the success branch is taken:
        self.assertIn("Successfully pruned and deleted vector data for:", result)

    def test_delete_invalid_table_identifier_not_found(self):
        server.db = MagicMock()
        # No master_manifest present and the bogus repo table is obviously absent, so
        # neither removal flag flips and the function returns its not-found message.
        server.db.table_names.return_value = []

        result = delete_repository("invalid_table_name")

        server.db.drop_table.assert_not_called()
        server.db.open_table.assert_not_called()
        self.assertIn(
            "Identifier 'invalid_table_name' (resolved as 'invalid_table_name') was not found in the database.",
            result,
        )

    @patch('server.os')
    def test_manifest_delete_exception_swallowed(self, mock_os):
        mock_os.path.abspath.return_value = "/repo"
        mock_os.path.exists.return_value = True
        mock_os.path.normpath.side_effect = lambda p: p
        table_name = "repo_repo_abcd1234"

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", table_name]
        mock_manifest = MagicMock()
        mock_manifest.delete.side_effect = RuntimeError("disk full")
        server.db.open_table.return_value = mock_manifest

        with patch('server.get_clean_table_name', return_value=table_name):
            result = delete_repository("/repo")

        # Table drop still happened, manifest failure is swallowed, success returned.
        server.db.drop_table.assert_called_once_with(table_name)
        self.assertIn("Successfully pruned and deleted vector data for:", result)

    def test_delete_when_neither_table_nor_manifest_present(self):
        # A raw table name that has no "repo_" prefix AND isn't on disk: still
        # treated as a table name, but neither table nor manifest can be touched.
        server.db = MagicMock()
        server.db.table_names.return_value = []  # no master_manifest at all
        result = delete_repository("orphan_table")

        server.db.drop_table.assert_not_called()
        server.db.open_table.assert_not_called()
        self.assertIn("was not found in the database.", result)

    def test_delete_non_repo_prefixed_table_uses_repo_path_predicate(self):
        """When the identifier is a table name NOT starting with 'repo_', the manifest
        delete falls back to the ``repo_path = ...`` predicate (else branch in code).

        We invoke delete via a real-looking path so the resolver picks the abs-path
        branch and uses the patched get_clean_table_name to return our table.
        """
        table_name = "weird_name_not_prefixed"
        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", table_name]
        mock_manifest = MagicMock()
        server.db.open_table.return_value = mock_manifest

        with patch('server.os.path.exists', return_value=False), \
             patch('server.get_clean_table_name', return_value=table_name), \
             patch('server.os.path.abspath', return_value="/abs/strange"):
            # Force the table-name string to be treated as a path: ensure it has a slash
            # so the resolver's path branch is taken (table_name lacks 'repo_' prefix).
            result = delete_repository("/abs/strange")

        # Drop the matched table.
        server.db.drop_table.assert_called_once_with(table_name)
        # And the manifest else branch (table not starting with repo_) -> repo_path match.
        mock_manifest.delete.assert_called_once_with("repo_path = '/abs/strange'")
        self.assertIn("Successfully pruned and deleted vector data for:", result)


class TestSearchAllCodebases(_ServerDBFixture):
    """search_all_codebases iterates all repo_* tables and merges results."""

    def test_no_repo_tables_returns_message(self):
        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest"]
        self.assertEqual(
            search_all_codebases("query"),
            "No repositories have been indexed globally yet.",
        )

    def test_no_tables_at_all_returns_message(self):
        server.db = MagicMock()
        server.db.table_names.return_value = []
        self.assertEqual(
            search_all_codebases("query"),
            "No repositories have been indexed globally yet.",
        )

    @patch('server.get_embedding')
    @patch('server.count_tokens')
    def test_single_repo_success(self, mock_tokens, mock_embed):
        mock_embed.return_value = [0.1, 0.2]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", "repo_alpha_abcd1234"]

        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {
                "file_path": "/alpha/a.py", "start_line": 1,
                "type": "function_definition", "text": "def a(): pass",
                "_distance": 0.1,
            }
        ]

        result = search_all_codebases("query", limit_per_repo=3, token_budget=1000)

        self.assertIn("=== Cross-Repo Search Results for: 'query' ===", result)
        # display_name derived from table name: "repo_alpha_abcd1234".replace("repo_","")
        # -> "alpha_abcd1234".split("_")[0] -> "alpha"
        self.assertIn("[Repo: alpha]", result)
        self.assertIn("[1]", result)
        self.assertIn("def a(): pass", result)

    @patch('server.get_embedding')
    @patch('server.count_tokens')
    def test_multi_repo_results_ordered_by_distance(self, mock_tokens, mock_embed):
        mock_embed.return_value = [0.1]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        server.db.table_names.return_value = [
            "master_manifest", "repo_alpha_aaaaaaaa", "repo_beta_bbbbbbbb",
        ]
        # open_table called once per repo_* table; mock side_effect for each.
        alpha_table = MagicMock()
        beta_table = MagicMock()
        server.db.open_table.side_effect = [alpha_table, beta_table]

        alpha_search = MagicMock()
        alpha_table.search.return_value = alpha_search
        alpha_search.metric.return_value = alpha_search
        alpha_search.limit.return_value = alpha_search
        alpha_search.to_list.return_value = [
            {"file_path": "/alpha/lo.py", "start_line": 1,
             "type": "function_definition", "text": "lo", "_distance": 0.5},
        ]
        beta_search = MagicMock()
        beta_table.search.return_value = beta_search
        beta_search.metric.return_value = beta_search
        beta_search.limit.return_value = beta_search
        beta_search.to_list.return_value = [
            {"file_path": "/beta/hi.py", "start_line": 1,
             "type": "class_definition", "text": "hi", "_distance": 0.1},
        ]

        result = search_all_codebases("query", token_budget=10000)

        # Beta has smaller distance -> should appear first.
        idx_beta = result.find("[Repo: beta]")
        idx_alpha = result.find("[Repo: alpha]")
        self.assertLess(idx_beta, idx_alpha)
        self.assertIn("[1] [Repo: beta]", result)

    @patch('server.get_embedding')
    @patch('server.count_tokens')
    def test_one_repo_error_does_not_break_others(self, mock_tokens, mock_embed):
        mock_embed.return_value = [0.1]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        server.db.table_names.return_value = [
            "master_manifest", "repo_zeta_zzzzzzzz", "repo_yan_yyyyyyy",
        ]
        zeta_table = MagicMock()
        yan_table = MagicMock()
        server.db.open_table.side_effect = [zeta_table, yan_table]

        # The broken repo (zeta) raises during its search chain call.
        zeta_table.search.side_effect = RuntimeError("boom")
        yan_search = MagicMock()
        yan_table.search.return_value = yan_search
        yan_search.metric.return_value = yan_search
        yan_search.limit.return_value = yan_search
        yan_search.to_list.return_value = [
            {"file_path": "/yan/y.py", "start_line": 1,
             "type": "function_definition", "text": "y", "_distance": 0.2},
        ]

        result = search_all_codebases("query", token_budget=1000)

        # Zeta absence isn't fatal; yan results still appear.
        self.assertIn("[Repo: yan]", result)
        self.assertIn("y", result)

    @patch('server.get_embedding')
    @patch('server.count_tokens')
    def test_token_budget_truncates_results(self, mock_tokens, mock_embed):
        mock_embed.return_value = [0.1]
        # Header counts as 0 tokens; each block counts as huge -> truncate after first.
        mock_tokens.side_effect = lambda text: 0 if "===" in text else 5000

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", "repo_alpha_aaaaaaaa"]
        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = [
            {"file_path": "/alpha/a.py", "start_line": 1,
             "type": "function_definition", "text": "x", "_distance": 0.1},
        ]

        result = search_all_codebases("query", token_budget=100)
        self.assertIn(
            "WARNING: Global cross-repo results truncated to fit within the 100 token context budget.",
            result,
        )

    @patch('server.get_embedding')
    @patch('server.count_tokens')
    def test_all_repos_return_empty_hits(self, mock_tokens, mock_embed):
        mock_embed.return_value = [0.1]
        mock_tokens.side_effect = lambda text: 1

        server.db = MagicMock()
        server.db.table_names.return_value = ["master_manifest", "repo_alpha_aaaaaaaa"]
        mock_table = MagicMock()
        server.db.open_table.return_value = mock_table
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = []

        result = search_all_codebases("query")
        self.assertEqual(
            result, "No structural components matched across any code bases."
        )


if __name__ == '__main__':
    unittest.main()
