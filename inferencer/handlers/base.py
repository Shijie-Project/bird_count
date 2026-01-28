class BaseResultHandler:
    """Base class for result handlers."""

    def setup(self):
        pass

    def cleanup(self):
        pass

    def handle(self, sids, processed_images, counts, timestamp):
        raise NotImplementedError
