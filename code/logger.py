import logging
from datetime import datetime
import os
import sys

class Logger:
    def __init__(self, log_dir="logs", log_filename=None):
        os.makedirs(log_dir, exist_ok=True)
        if not log_filename:
            log_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 중복 핸들러 방지
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.logger.info("Logger initialized")

    def log_user_input(self, message: str):
        try:
            self.logger.info(f"[User] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode("utf-8", "ignore").decode()
            self.logger.info(f"[User] {safe_message}")

    def log_assistant_response(self, message: str):
        try:
            self.logger.info(f"[Assistant] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode("utf-8", "ignore").decode()
            self.logger.info(f"[Assistant] {safe_message}")

    def log_system(self, message: str):
        try:
            self.logger.info(f"[System] {message}")
        except UnicodeEncodeError:
            safe_message = message.encode("utf-8", "ignore").decode()
            self.logger.info(f"[System] {safe_message}")