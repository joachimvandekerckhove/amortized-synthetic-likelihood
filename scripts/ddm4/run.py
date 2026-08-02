#!/usr/bin/env python3
"""Application entry point for the ddm4 ASL pipeline."""

import sys

from models.ddm.ddm4 import DDM4

STEPS = ("generate-data", "train-emulator", "wire-to-jags", "confirm-recovery")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(STEPS)}>", file=sys.stderr)
        sys.exit(1)

    step = sys.argv[1]
    if step == "generate-data":
        from asl.cov_data import generate_cov_dataset

        generate_cov_dataset(DDM4)
    elif step == "train-emulator":
        from asl.train import train_emulator

        train_emulator(DDM4)
    elif step == "wire-to-jags":
        from asl.wire import wire_to_jags

        wire_to_jags(DDM4)
    elif step == "confirm-recovery":
        from asl.recovery import run_recovery_study

        run_recovery_study(DDM4)


if __name__ == "__main__":
    main()
