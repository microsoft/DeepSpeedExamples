from enum import Enum

class Status(str, Enum):
    OK = "OK"
    FAIL = "FAIL"

    def __str__(self) -> str:
        return self.value