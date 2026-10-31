class IndexingError(ValueError):
    """An indexing error suitable for display without exposing library traces."""


class ModelLoadingError(IndexingError):
    """The shared model is unavailable; do not retry its download for every file."""
