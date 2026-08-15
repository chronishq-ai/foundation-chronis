# Repository Architecture

```text
chronis-ml/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
├── src/
│   └── chronis_ml/
├── tests/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── pyproject.toml
```

The `src/` layout prevents accidental imports from the repository root.

This Phase 1 scaffold intentionally keeps domain modules small until the
canonical data interfaces are agreed by the FOUNDRY team.
