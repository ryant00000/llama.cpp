#!/usr/bin/env python3
"""Extract LLAMA_API function signatures from include/llama.h.

Outputs one normalized signature per line, sorted alphabetically by function
name.  The result is suitable for committing as scripts/libllama.abi and for
diffing in CI to detect ABI changes.

Usage:
    python3 scripts/gen-libllama-abi.py [path/to/llama.h]
"""

import re
import sys


def preprocess(text: str) -> str:
    """Strip comments and preprocessor directives, keeping newlines for
    accurate error reporting (we don't use line numbers here but it keeps
    the character offsets meaningful for debugging)."""

    # Remove /* ... */ block comments (may span lines).
    text = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group().count('\n'), text, flags=re.DOTALL)

    # Remove // ... line comments (keep the newline).
    text = re.sub(r'//[^\n]*', '', text)

    # Remove preprocessor directive lines (lines where the first non-space
    # char is '#').  Replace with blank lines to preserve offsets.
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        if line.lstrip().startswith('#'):
            result.append('\n' * line.count('\n'))
        else:
            result.append(line)
    return ''.join(result)


def normalize(s: str) -> str:
    """Collapse all whitespace runs to a single space and strip edges."""
    return re.sub(r'\s+', ' ', s).strip()


def extract_signatures(header_text: str) -> list[str]:
    """Return a sorted list of normalized LLAMA_API function signatures."""

    text = preprocess(header_text)
    sigs: list[str] = []

    i = 0
    n = len(text)

    while i < n:
        # Find the next LLAMA_API token.
        pos = text.find('LLAMA_API', i)
        if pos == -1:
            break
        i = pos + len('LLAMA_API')

        # Skip leading whitespace after LLAMA_API.
        while i < n and text[i] in ' \t\r\n':
            i += 1

        # Determine whether we are inside DEPRECATED(...).
        #
        # Case A: DEPRECATED(LLAMA_API ..., "hint");
        #   – look back before the LLAMA_API token for 'DEPRECATED('
        # Case B: LLAMA_API DEPRECATED(return_type func(...), "hint");
        #   – look forward for 'DEPRECATED('

        # Case A: look back (skip whitespace) for 'DEPRECATED('
        before = text[:pos].rstrip()
        in_deprecated_wrap = before.endswith('DEPRECATED(')

        if in_deprecated_wrap:
            # We are the argument list of DEPRECATED(LLAMA_API ..., "hint");
            # Collect everything until the matching ')' that closes DEPRECATED,
            # then strip the trailing , "hint" part.
            depth = 1   # we just entered DEPRECATED(
            start = i   # start of "return_type func_name(..."
            j = i
            while j < n and depth > 0:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            # text[start:j-1] is everything inside DEPRECATED(...).
            # We need the function signature part, which ends at the last
            # top-level comma (separating the function from the "hint" string).
            inner = text[start:j - 1]
            # Find the last top-level comma.
            depth2 = 0
            last_comma = -1
            for k, ch in enumerate(inner):
                if ch == '(':
                    depth2 += 1
                elif ch == ')':
                    depth2 -= 1
                elif ch == ',' and depth2 == 0:
                    last_comma = k
            sig_raw = inner[:last_comma] if last_comma != -1 else inner
            i = j
        elif text[i:i + len('DEPRECATED(')] == 'DEPRECATED(':
            # Case B: LLAMA_API DEPRECATED(return_type func(...), "hint");
            i += len('DEPRECATED(')
            depth = 1
            start = i
            j = i
            while j < n and depth > 0:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            inner = text[start:j - 1]
            depth2 = 0
            last_comma = -1
            for k, ch in enumerate(inner):
                if ch == '(':
                    depth2 += 1
                elif ch == ')':
                    depth2 -= 1
                elif ch == ',' and depth2 == 0:
                    last_comma = k
            sig_raw = inner[:last_comma] if last_comma != -1 else inner
            i = j
        else:
            # Plain: LLAMA_API return_type func_name(...);
            # Collect until the ';' at parenthesis depth 0.
            depth = 0
            start = i
            j = i
            while j < n:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                elif text[j] == ';' and depth == 0:
                    break
                j += 1
            sig_raw = text[start:j]
            # Advance past the ';'
            i = j
            while i < n and text[i] in ' \t\r\n;':
                i += 1

        sig = normalize(sig_raw)
        if sig and '(' in sig:
            sigs.append(sig)

    sigs.sort(key=lambda s: re.search(r'\b(llama_\w+)\s*\(', s).group(1) if re.search(r'\b(llama_\w+)\s*\(', s) else s)
    return sigs


def main() -> None:
    header_path = sys.argv[1] if len(sys.argv) > 1 else 'include/llama.h'
    with open(header_path, encoding='utf-8') as f:
        text = f.read()
    for sig in extract_signatures(text):
        print(sig)


if __name__ == '__main__':
    main()
