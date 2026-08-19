from __future__ import annotations

import argparse
import sys

from chronis_ml.ops import check_licenses, pip_audit, write_sbom
from chronis_ml.store import IsolationError, assert_src_isolated, root_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["isolation", "audit", "all"])
    args = p.parse_args(argv)
    root = root_dir()
    if args.cmd in {"isolation", "all"}:
        try:
            assert_src_isolated(root / "src")
        except IsolationError as e:
            print(e)
            return 1
        print("isolation ok")
    if args.cmd in {"audit", "all"}:
        check_licenses(root)
        write_sbom(root)
        code = pip_audit()
        if code:
            return code
        print("audit ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
