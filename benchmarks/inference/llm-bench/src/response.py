from dataclasses import dataclass

@dataclass
class Response:
    text: str
    request_time: float = 0