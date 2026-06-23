#!/usr/bin/env python3
"""Application entry point for the ddm4mv ESL pipeline."""

import sys

from esl.registry import register
from models.ddm.ddm4mv import DDM4MV

SLUG = "ddm4mv"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "generate-data":
        print(f"Usage: {sys.argv[0]} generate-data", file=sys.stderr)
        sys.exit(1)

    register(DDM4MV)
    from esl.cov_data import generate_cov_dataset

    generate_cov_dataset(SLUG)


if __name__ == "__main__":
    main()
