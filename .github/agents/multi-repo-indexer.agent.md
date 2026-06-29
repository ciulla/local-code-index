---
name: Multi-Repo Indexer
description: This custom agent indexes multiple repositories and allows for cross-repository search.
argument-hint: Provide a list of repository paths to index.
target: vscode
model: GPT-4.1
tools: [execute, read, agent, search, 'multi-repo-indexer/*']
---