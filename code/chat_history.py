class ChatHistory:
    def __init__(self, history_limit: int = 5):
        self.history_limit = history_limit
        self.history = []

    def add_user_message(self, message: str):
        self._add({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self._add({"role": "assistant", "content": message})

    def _add(self, message):
        self.history.append(message)
        # 히스토리 길이 제한 유지 (user + assistant 메시지로 구성되므로 *2)
        if len(self.history) > self.history_limit * 2:
            self.history.pop(0)

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

    def __repr__(self):
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)
