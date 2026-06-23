"""
esl.registry -- Model registry.

Application scripts register models before calling ESL pipeline functions.
The registry is intentionally empty at import time so esl stays model-agnostic.
"""

from esl.spec import Model

_REGISTRY: dict[str, Model] = {}


def register(model: Model) -> None:
    """Add a model to the global registry.

    Parameters
    ----------
    model : Model
        A fully specified Model instance.
    """
    _REGISTRY[model.slug] = model


def get_model(slug: str) -> Model:
    """Retrieve a registered model by slug.

    Parameters
    ----------
    slug : str
        The model identifier (e.g., "ddm3" or "ddm4").

    Returns
    -------
    Model

    Raises
    ------
    KeyError
        If the slug has not been registered.
    """
    if slug not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown model slug '{slug}'. Available: {available}. "
            "Register the model in your application script before calling ESL."
        )
    return _REGISTRY[slug]


def registered_slugs() -> list[str]:
    """Return sorted list of registered model slugs."""
    return sorted(_REGISTRY.keys())
