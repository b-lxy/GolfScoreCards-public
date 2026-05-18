from   pathlib    import Path

PROJECT_ROOT = Path().resolve().parent

ALPHABET = (
    "A-Z, 0-9\n"
    "Symbols: @ : Circle, ^ : Triangle, + : Square, | : Delimiter"
)

TABLE_LINE_INTENSITY = 220

BOLD_L = "\033[1m"
BOLD_R = "\033[0m"