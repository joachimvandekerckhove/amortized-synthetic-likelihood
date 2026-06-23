#!/usr/bin/env python3
"""Application entry point for the ddm3mv ESL pipeline."""

import sys

from esl.registry import register
from models.ddm.ddm3mv import DDM3MV

SLUG = "ddm3mv"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "generate-data":
        print(f"Usage: {sys.argv[0]} generate-data", file=sys.stderr)
        sys.exit(1)

    register(DDM3MV)
    from esl.cov_data import generate_cov_dataset

    generate_cov_dataset(SLUG)


if __name__ == "__main__":
    main()
