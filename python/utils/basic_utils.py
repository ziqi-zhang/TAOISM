import hashlib
from enum import Enum

def str_hash(s):
    return int(int(hashlib.sha224(s.encode('utf-8')).hexdigest(), 16) % ((1 << 62) - 1))

class ExecutionModeOptions(Enum):
    Enclave = 1
    CPU = 2
    GPU = 3