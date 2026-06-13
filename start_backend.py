"""Start the FastAPI backend. Run from the project root (CLNP/).

Usage:
    python start_backend.py
    python start_backend.py --port 8000
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not found. Run: pip install uvicorn[standard]")
        sys.exit(1)

    # Pass config path via environment so the app picks it up on import
    import os
    os.environ["CLNP_CONFIG"] = args.config

    print(f"\n🚀  YAKE + LLM Pipeline API")
    print(f"   Config : {args.config}")
    print(f"   Docs   : http://{args.host}:{args.port}/docs")
    print(f"   Ctrl+C to stop\n")

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()