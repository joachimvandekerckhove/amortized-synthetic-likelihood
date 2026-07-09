#!/usr/bin/env python3
"""Application entry point for the ddm3mv ASL pipeline."""

import sys

from asl.registry import register
from models.ddm.ddm3mv import DDM3MV

SLUG = "ddm3mv"


def main() -> None:
    steps = ("generate-data", "train-emulator", "wire-to-jags", "confirm-recovery")
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        print(f"Usage: {sys.argv[0]} <{'|'.join(steps)}>", file=sys.stderr)
        sys.exit(1)

    register(DDM3MV)
    step = sys.argv[1]

    if step == "generate-data":
        from asl.cov_data import generate_cov_dataset

        generate_cov_dataset(SLUG)
    elif step == "train-emulator":
        from asl.train_mv import train_emulator_mv

        train_emulator_mv(SLUG)
    elif step == "wire-to-jags":
        from asl.wire import wire_to_jags

        wire_to_jags(SLUG)
    elif step == "confirm-recovery":
        from asl.recovery_mv import run_recovery_study_mv

        run_recovery_study_mv(SLUG)


if __name__ == "__main__":
    main()
