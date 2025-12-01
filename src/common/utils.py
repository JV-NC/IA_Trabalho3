from typing import Dict

RESET = "\033[0m"
BOLD = "\033[1m"

# Cores
COMPONENT = "\033[96m" #CYAN
TITLE = "\033[95m" #MAGENTA
PRICE = "\033[92m" #GREEN
BORDER = "\033[94m" #BLUE
CHOICE = "\033[97m" #WHITE

def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"

def print_build(build) -> None:
    """
    Show final build table with colors.
    """

    title_line = "=" * 29
    print(
        "\n"
        + color(title_line, BORDER)
        + " "
        + color("FINAL BUILD", TITLE + BOLD)
        + " "
        + color(title_line, BORDER)
    )

    #Header
    header = (
        f"{color('Component', TITLE):20} | "
        f"{color('Choice', TITLE):35} | "
        f"{color('Price (R$)', TITLE)}"
    )
    print(header)

    print(color("-" * 71, BORDER))

    # Table body
    total = 0
    for comp, (name, price) in build.items():
        total += price
        print(
            f"{color(comp, COMPONENT):20} | "
            f"{color(name, CHOICE):35} | "
            f"{color(f'{price:>8}', PRICE)}"
        )

    print(color("-" * 71, BORDER))

    # Total
    print(
        f"{color('TOTAL', TITLE + BOLD):20} | "
        f"{'':35} | "
        f"{color(f'{total:>8}', PRICE + BOLD)}"
    )

    print(color("=" * 71, BORDER) + "\n")