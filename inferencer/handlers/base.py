from ..types import ResultItem


class BaseResultHandler:
    """Base class for all result processing modules."""

    def setup(self):
        """Optional initialization logic."""
        pass

    def cleanup(self):
        """Optional cleanup logic."""
        pass

    def handle(self, results: ResultItem):
        """Main processing logic to be implemented by subclasses."""
        raise NotImplementedError
