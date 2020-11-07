import logging
import sys


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")


def setup_logger(out_file=None, stdout=True, stdout_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stdout_level, file_level))

    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stdout_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER
