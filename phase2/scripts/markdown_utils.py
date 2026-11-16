def join_paragraph_lines(text):
    # Satu paragraf satu baris; tabel, daftar, dan blok kode tetap dipisah.
    result, paragraph = [], []
    in_code = False

    def flush():
        if paragraph:
            result.append(" ".join(paragraph))
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            result.append(line)
        elif in_code:
            result.append(line)
        elif not stripped or stripped.startswith(("#", "|", "<!--", "- ", "* ", ">")):
            flush()
            result.append(line)
        elif line.startswith("  ") and result and result[-1].lstrip().startswith(("- ", "* ")):
            result[-1] += " " + stripped
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(result) + "\n"
