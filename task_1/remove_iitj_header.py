import argparse
import re
from collections import Counter
from pathlib import Path


KNOWN_NAV_PHRASES = {
    "home",
    "about iitj",
    "institute",
    "statutory bodies",
    "key functionaries",
    "infrastructure",
    "outreach",
    "administration",
    "office of director",
    "office of deputy director",
    "office of registrar",
    "office of administration",
    "various offices",
    "administrative contact",
    "academics",
    "departments",
    "schools",
    "centers",
    "idrps/idrcs",
    "people",
    "recruitment",
    "about recruitment",
    "faculty members",
    "post doctoral fellows",
    "technical staff members",
    "project staff members",
    "research",
    "publication",
    "about research",
    "ongoing projects",
    "completed projects",
    "collaborations",
    "academic programs",
    "undergraduate program",
    "postgraduate program",
    "doctorial program",
    "doctoral program",
    "doctoral programs",
    "curriculum",
    "courses",
    "program structure",
    "regulations",
    "laboratories",
    "contact",
    "previous",
    "next",
    "play_arrow",
    "highlights",
    "view all highlights",
    "announcement",
    "view all announcement",
    "sitemap",
}


def normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"\s+", " ", line).strip().lower()
    line = line.strip("|:-_.,;()[]{}")
    return line


def looks_like_ui_noise(normalized_line: str) -> bool:
    if not normalized_line:
        return True

    if normalized_line in {"a+", "a", "a-", "hindi", "hindi |", "| hindi", "hindi | sitemap"}:
        return True

    if "redirecttologinpage" in normalized_line:
        return True

    if re.fullmatch(r"#+", normalized_line):
        return True

    return False


def build_boilerplate_set(files, min_doc_ratio: float, max_tokens: int):
    doc_freq = Counter()

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        seen_in_doc = set()
        for raw_line in text.splitlines():
            norm = normalize_line(raw_line)
            if not norm:
                continue

            token_count = len(norm.split())
            if token_count == 0 or token_count > max_tokens:
                continue

            if len(norm) > 90:
                continue

            seen_in_doc.add(norm)

        doc_freq.update(seen_in_doc)

    total_docs = max(1, len(files))
    min_docs = max(2, int(total_docs * min_doc_ratio))

    boilerplate = {line for line, freq in doc_freq.items() if freq >= min_docs}
    boilerplate.update(KNOWN_NAV_PHRASES)
    return boilerplate


def clean_single_text(text: str, boilerplate_set):
    cleaned_lines = []
    prev_norm = None

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.replace("\xa0", " ")).strip()
        norm = normalize_line(line)

        if not norm:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if looks_like_ui_noise(norm):
            continue

        if norm in boilerplate_set:
            continue

        # Drop tiny menu-like fragments that survive frequency filtering.
        if len(norm.split()) <= 2 and len(norm) <= 20:
            if norm in {"search", "menu", "login", "skip to content", "quick links"}:
                continue

        if prev_norm == norm:
            continue

        cleaned_lines.append(line)
        prev_norm = norm

    # Remove leading/trailing blank lines and collapse repeated blanks.
    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    final_lines = []
    for line in cleaned_lines:
        if line == "" and final_lines and final_lines[-1] == "":
            continue
        final_lines.append(line)

    return "\n".join(final_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Remove repeated IITJ navigation/header text from web extracted documents."
    )
    parser.add_argument("--input", default="data/web", help="Folder with raw text files")
    parser.add_argument("--output", default="data/web_clean", help="Folder for cleaned text files")
    parser.add_argument(
        "--min-doc-ratio",
        type=float,
        default=0.30,
        help="Line seen in at least this ratio of docs is treated as boilerplate (default: 0.30)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Only lines up to this token count are considered for boilerplate detection",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    boilerplate_set = build_boilerplate_set(files, args.min_doc_ratio, args.max_tokens)
    print(boilerplate_set)
    written = 0
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_single_text(text, boilerplate_set)

        out_path = output_dir / file_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        written += 1

    print(f"Processed {written} files")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Boilerplate line count: {len(boilerplate_set)}")


if __name__ == "__main__":
    main()
