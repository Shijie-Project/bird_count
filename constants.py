import os
from dataclasses import dataclass


def str_to_bool(v):
    return v.lower() in ("yes", "true", "t", "1")


@dataclass(frozen=True)
class EVN:
    debug: bool = str_to_bool(os.getenv("DEBUG", "False"))
