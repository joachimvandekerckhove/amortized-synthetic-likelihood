"""Lookup published models by slug (used by multiprocessing workers)."""

from asl.spec import Model


def get_model(slug: str) -> Model:
    """Return a published model by slug."""
    from models.ddm.ddm3 import DDM3
    from models.ddm.ddm4 import DDM4
    from models.ddm.ddmcollapsesig import DDMCOLLAPSESIG
    from models.social.dw import DW

    by_slug = {
        DDM3.slug: DDM3,
        DDM4.slug: DDM4,
        DDMCOLLAPSESIG.slug: DDMCOLLAPSESIG,
        DW.slug: DW,
    }
    if slug not in by_slug:
        available = ", ".join(sorted(by_slug))
        raise KeyError(f"Unknown model slug '{slug}'. Available: {available}")
    return by_slug[slug]
