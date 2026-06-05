import logging
import sys


def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "%(message)s\n"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger