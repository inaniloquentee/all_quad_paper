import re
import zlib
from pathlib import Path


def main() -> None:
    pdf = Path("all_quad_paper.pdf").read_bytes()
    for match in re.finditer(
        rb"(\d+)\s+(\d+)\s+obj(.*?)stream\r?\n(.*?)\r?\nendstream",
        pdf,
        re.S,
    ):
        obj = int(match.group(1))
        header = match.group(3)
        data = match.group(4)
        if b"/FlateDecode" not in header:
            continue
        try:
            decoded = zlib.decompress(data)
        except zlib.error:
            continue
        if not any(
            token in decoded
            for token in (b"BT", b"Tj", b"TJ", b"/ToUnicode", b"beginbfchar", b"beginbfrange")
        ):
            continue
        print(f"\n=== OBJ {obj} LEN {len(decoded)} ===")
        print(header[:240].decode("latin1", "replace"))
        print(decoded[:3000].decode("latin1", "replace"))


if __name__ == "__main__":
    main()
