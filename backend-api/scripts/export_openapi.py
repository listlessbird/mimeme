from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("PRELOAD_TEXT_ENCODER_ON_STARTUP", "false")

from api.main import create_app  # noqa: E402


def main() -> None:
    schema = create_app().openapi()
    payload = json.dumps(schema, indent=2)

    if len(sys.argv) > 1:
        out_path = sys.argv[1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
