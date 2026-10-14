"""Tabelas de texto e Markdown dos relatórios (`python -m strategy.<módulo>`)."""


def render_text(table: list[list[str]]) -> str:
    """Tabela alinhada para o terminal; a primeira linha é o cabeçalho."""
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render_markdown(table: list[list[str]]) -> str:
    head, *body = table
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def pct(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}%"
