"""Backwards-compatible wrapper around the CLI pipeline."""
from mobile_money_project.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
