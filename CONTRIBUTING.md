# Contributing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
crisis-vision demo --output artifacts/smoke --frames 12 --no-video
```

## Requirements

- Keep imports free of video I/O and GUI side effects.
- Put configurable thresholds in `CrisisConfig`.
- Add tests for rule, geometry, tracker, or contract changes.
- Preserve JSON compatibility or deliberately version the schema.
- Label synthetic and real evaluation results separately.
- Document source and license before adding any template or footage.
- Never commit credentials, private incident footage, or personal data.

Pull requests should include the commands run, a brief risk/limitation note, and any
observable output change.
