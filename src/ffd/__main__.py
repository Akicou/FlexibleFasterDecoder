"""Entry point for `python -m ffd`."""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
