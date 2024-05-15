from enum import Enum


# TODO: Expand on Status codes and make useful within the benchmark (e.g., better error messages)
class Status(str, Enum):
    OK = "OK"
    FAIL = "FAIL"

    def __str__(self) -> str:
        return self.value
