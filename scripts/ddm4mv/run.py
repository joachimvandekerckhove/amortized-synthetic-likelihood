#!/usr/bin/env python3
"""Application entry point for the ddm4mv ESL pipeline."""

import sys

from esl.registry import register
from models.ddm.ddm4mv import DDM4MV

SLUG = "ddm4mv"


def main() -> None:
    steps = ("generate-data", "train-emulator", "wire-to-jags", "confirm-recovery")
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        print(f"Usage: {sys.argv[0]} <{'|'.join(steps)}>", file=sys.stderr)
        sys.exit(1)

    register(DDM4MV)
    step = sys.argv[1]

    if step == "generate-data":
        from esl.cov_data import generate_cov_dataset

        generate_cov_dataset(SLUG)
    elif step == "train-emulator":
        from esl.train_mv import train_emulator_mv

        train_emulator_mv(SLUG)
    elif step == "wire-to-jags":
        from esl.wire import wire_to_jags

        wire_to_jags(SLUG)
    elif step == "confirm-recovery":
        from esl.recovery_mv import run_recovery_study_mv

        run_recovery_study_mv(SLUG)


if __name__ == "__main__":
    main()
