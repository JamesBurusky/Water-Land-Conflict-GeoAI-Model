"""
encoding_check.py

Run this against any file that throws UnicodeDecodeError to (a) see what
character is actually at the failing byte offset, and (b) get a
best-guess encoding via charset-normalizer, before deciding how to read it.

Usage:
    python src/encoding_check.py data/your_file.csv
"""
import sys

def inspect(path: str, error_position: int | None = None):
    with open(path, "rb") as f:
        raw = f.read()

    print(f"File size: {len(raw):,} bytes")

    if error_position is not None:
        start = max(0, error_position - 40)
        end = min(len(raw), error_position + 40)
        context = raw[start:end]
        print(f"\nBytes around position {error_position}:")
        print(context)
        # Show how cp1252 and latin-1 each interpret it
        try:
            print("\nAs cp1252:  ", context.decode("cp1252"))
        except Exception as e:
            print("cp1252 failed:", e)
        try:
            print("As latin-1: ", context.decode("latin-1"))
        except Exception as e:
            print("latin-1 failed:", e)

    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best:
            print(f"\ncharset-normalizer best guess: {best.encoding} "
                  f"(confidence-ish; not exact science)")
    except ImportError:
        print("\n(install charset-normalizer for an automatic guess: "
              "pip install charset-normalizer)")

if __name__ == "__main__":
    path = sys.argv[1]
    pos = int(sys.argv[2]) if len(sys.argv) > 2 else None
    inspect(path, pos)
