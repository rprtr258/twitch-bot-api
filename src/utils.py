from dataclasses import dataclass
from typing import Optional


@dataclass
class MessageRecord:
    username: str
    channel: str
    message: Optional[str]

    def __str__(self) -> str:
        return f"[{self.channel}] {self.username}: {self.message}"
