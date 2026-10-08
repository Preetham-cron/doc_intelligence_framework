import logging
import os
import time

import mysql.connector
from dotenv import load_dotenv

# Load from .envs/.local first (local dev), fall back to .env
if os.path.exists(".envs/.local"):
    load_dotenv(dotenv_path=".envs/.local", override=True)
else:
    load_dotenv(dotenv_path=".env")

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Log to console
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def connect_to_mysql(attempts=2, delay=2):
    attempt = 1

    config = {
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "host": os.environ.get("DB_HOST"),
        "database": os.environ.get("DB_NAME"),
        "raise_on_warnings": True,
        "port": os.environ.get("DB_PORT") or 3306,
	  "use_pure": True,
    }

        # Implement a reconnection routine
    while attempt < attempts + 1:
        try:
            return mysql.connector.connect(**config)
        except (mysql.connector.Error, IOError) as err:
            if attempt == attempts:
                # Attempts to reconnect failed; returning None
                logger.error("Failed to connect, exiting without a connection: %s", err)
                return None
            logger.error(
                "Connection failed: %s. Retrying (%d/%d)...",
                err,
                attempt,
                attempts - 1,
            )
            # progressive reconnect delay
            time.sleep(delay**attempt)
            attempt += 1
    return None
