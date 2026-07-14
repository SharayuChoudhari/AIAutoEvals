# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog; adheres to Semantic Versioning.

## [Unreleased]

## [0.1.4] - 2026-07-14
### Fixed
- `ai-evals` CLI failed to check local imports due to incorrect current working directory handling.

## [0.1.0] - 2026-07-10
### Added
- Phase 1: static analysis & scaffolding (`ai-evals init`, `analyze`, `doctor`, `config`).
- Phase 2: model-agnostic judge gateway via LiteLLM + Instructor (`ai-evals judge`).
- Phase 3: automated golden-set bootstrapper (`ai-evals bootstrap`).
- Phase 4: run execution & insights (`ai-evals run`, `diff`, `report`, `history`).
