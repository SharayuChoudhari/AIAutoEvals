# Releasing `dot-ai-evals` to GitHub & PyPI

This repo uses **tag-triggered GitHub Actions** to build and publish to PyPI.
There is no manual `twine upload` — publishing uses **OIDC trusted publishing**
(no API tokens to rotate).

The pipeline lives at `.github/workflows/release.yml`.

---

## 0. Prerequisites (one-time setup)

### GitHub
1. Push the repo to GitHub (if not already):
   ```bash
   git remote add origin git@github.com:<user>/AIAutoEvals.git
   git push -u origin main
   ```
2. In the GitHub repo, go to **Settings → Environments** and create two
   environments:
   - `testpypi`
   - `pypi`

### PyPI (OIDC trusted publishing)
Configure trusted publishing on both PyPI and TestPyPI so GitHub Actions can
publish without tokens.

**PyPI** (https://pypi.org):
1. Register the project `dot-ai-evals` (first release only; run a manual
   upload once if the name isn't claimed yet).
2. Go to **Account settings → Publishing → Add a new publisher**.
3. Choose **GitHub**, then fill in:
   - PyPI Project name: `dot-ai-evals`
   - Owner: `<your-github-user-or-org>`
   - Repository name: `AIAutoEvals`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

**TestPyPI** (https://test.pypi.org): repeat the same steps with environment
name `testpypi`.

> Without this, the `publish-pypi` / `publish-testpypi` jobs will fail with an
> OIDC error.

---

## 1. Bump the version

The version is declared in **`pyproject.toml`**:

```toml
[project]
name = "dot-ai-evals"
version = "0.1.3"   # ← bump this
```

Pick a [SemVer](https://semver.org/) bump:
- Patch (`0.1.3 → 0.1.4`): bug fixes only.
- Minor (`0.1.3 → 0.2.0`): new backwards-compatible features.
- Major (`0.1.3 → 1.0.0`): breaking changes.

Edit the line, e.g. for a patch release:

```bash
# using sed, or just edit the file
sed -i '' 's/^version = "0.1.3"/version = "0.1.4"/' pyproject.toml
```

Update `CHANGELOG.md` with the changes for the new version.

Stage and commit:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v0.1.4"
```

---

## 2. Push to GitHub

```bash
git push origin main
```

This updates `main` but does **not** trigger a release. Releases are triggered
by tags (see the `on.push.tags` trigger in `release.yml`).

---

## 3. Create and push the release tag

The tag **must** match the pattern `v*` (e.g. `v0.1.4`) for the workflow to
fire. The tag's version should equal the `version` in `pyproject.toml`.

```bash
git tag v0.1.4
git push origin v0.1.4
```

Pushing the tag kicks off the **Release** workflow:

1. **build** — `uv build` produces `dist/*.tar.gz` + `dist/*.whl`, then
   `twine check dist/*` validates metadata.
2. **publish-testpypi** — uploads to TestPyPI (OIDC trusted publishing).
3. **publish-pypi** — uploads to real PyPI (OIDC trusted publishing).

Watch progress: **Actions tab → Release**.

---

## 4. Verify the release

After the workflow turns green:

```bash
# install the new version from PyPI
pip install --upgrade dot-ai-evals
python -c "import ai_eval; print(ai_eval.__version__)" 2>/dev/null || pip show dot-ai-evals

# or check TestPyPI first (useful for dry runs)
pip install -i https://test.pypi.org/simple/ dot-ai-evals
```

View it on the web:
- https://pypi.org/project/dot-ai-evals/
- https://test.pypi.org/project/dot-ai-evals/

---

## 5. (Optional) GitHub Release

Tags alone are fine, but a GitHub Release gives users release notes:

```bash
gh release create v0.1.4 \
  --title "v0.1.4" \
  --notes-file CHANGELOG.md \
  --verify-tag
```

Or use the GitHub UI: **Releases → Draft a new release → choose tag `v0.1.4`**.

---

## Manual / dry-run publishing

The workflow also supports `workflow_dispatch` for manual runs without making a
tag — useful for testing the pipeline:

1. Go to **Actions → Release → Run workflow**.
2. Choose the `repository` input:
   - `testpypi` → publishes to TestPyPI only.
   - `pypi` → publishes to real PyPI only.

> `workflow_dispatch` skips the tag requirement, so be careful with `pypi`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `publish-pypi` fails with OIDC / permission error | Trusted publishing not configured on PyPI for environment `pypi`. Re-do step 0. |
| `twine check` fails on metadata | Check `pyproject.toml` — missing `readme`, bad `license-files`, etc. |
| Workflow doesn't trigger on tag push | Tag must start with `v` (e.g. `v0.1.4`), not `0.1.4`. |
| Wrong version published | The tag version and `pyproject.toml` version must match. The build reads the version from `pyproject.toml`, not the tag. |
| `uv build` fails locally | Run `uv sync` first to ensure `hatchling` is available. |

---

## Quick reference (cheat sheet)

```bash
# 1. bump version in pyproject.toml + update CHANGELOG.md
# 2. commit & push
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v0.1.4"
git push origin main

# 3. tag & push the tag (triggers the CI pipeline → PyPI)
git tag v0.1.4
git push origin v0.1.4

# 4. verify
pip install --upgrade dot-ai-evals
```
