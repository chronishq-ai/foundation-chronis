# Development Workflow

1. Pull the latest `main`.
2. Create a feature branch.
3. Implement one focused change.
4. Add or update tests.
5. Run:
   - `poetry run ruff check .`
   - `poetry run mypy`
   - `poetry run pytest`
   - `poetry run pre-commit run --all-files`
6. Push the branch.
7. Open a pull request.
8. Get a second-engineer review.
9. Merge only after CI and review pass.

Do not self-merge.

## Data safety

Raw/decrypted sensor or audio data must not be committed to Git and must not
be written to ordinary project files as part of local development.
