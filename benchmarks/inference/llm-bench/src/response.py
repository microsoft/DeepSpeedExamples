from dataclasses import dataclass, asdict


@dataclass
class Response:
    text: str
    request_time: float = 0

    def to_dict(self) -> dict:
        return asdict(self)
