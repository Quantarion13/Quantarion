import time
import json

class Metrics:
    def __init__(self):
        self.data = []

    def log(self, stage, latency_ms, hit=False):
        self.data.append({"stage":stage, "latency":latency_ms, "hit":hit})

    def dump(self, path="metrics.json"):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
