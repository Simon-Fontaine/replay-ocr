import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
