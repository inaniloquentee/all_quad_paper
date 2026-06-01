import re
import sys
import zlib
from pathlib import Path


def decode_pdf_literal(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != 0x5C:
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            break
        esc = raw[i]
        i += 1
        if esc in b"nrtbf":
            out.append({ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}[esc])
        elif esc in b"()\\":
            out.append(esc)
        elif 48 <= esc <= 55:
            digits = bytes([esc])
            for _ in range(2):
                if i < len(raw) and 48 <= raw[i] <= 55:
                    digits += bytes([raw[i]])
                    i += 1
                else:
                    break
            out.append(int(digits, 8) & 0xFF)
        elif esc in (10, 13):
            if esc == 13 and i < len(raw) and raw[i] == 10:
                i += 1
        else:
            out.append(esc)
    return out.decode("latin1", "replace")


def literal_strings(data: bytes):
    i = 0
    while i < len(data):
        if data[i] != 0x28:
            i += 1
            continue
        i += 1
        start = i
        depth = 1
        escaped = False
        while i < len(data) and depth:
            c = data[i]
            if escaped:
                escaped = False
            elif c == 0x5C:
                escaped = True
            elif c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    yield data[start:i]
                    i += 1
                    break
            i += 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pdf = Path("all_quad_paper.pdf").read_bytes()
    page = 0
    for match in re.finditer(
        rb"(\d+)\s+(\d+)\s+obj(.*?)stream\r?\n(.*?)\r?\nendstream",
        pdf,
        re.S,
    ):
        header = match.group(3)
        data = match.group(4)
        if b"/FlateDecode" not in header:
            continue
        try:
            decoded = zlib.decompress(data)
        except zlib.error:
            continue
        if not any(token in decoded for token in (b" BT", b"\nBT", b"Tj", b"TJ")):
            continue
        strings = [decode_pdf_literal(s) for s in literal_strings(decoded)]
        strings = [s for s in strings if any(ch.isalnum() for ch in s)]
        if not strings:
            continue
        page += 1
        print(f"\n\n=== TEXT STREAM {page} OBJ {int(match.group(1))} ===")
        line = []
        for s in strings:
            if len(s) == 1 and not s.isalnum():
                continue
            line.append(s)
            if len(" ".join(line)) > 110:
                print(" ".join(line))
                line = []
        if line:
            print(" ".join(line))


if __name__ == "__main__":
    main()
