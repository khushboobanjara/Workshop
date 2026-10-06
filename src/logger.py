import logging


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s -%(name)s -%(message)s"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

    return logger