import logging

def get_logger(name: str):
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(name)