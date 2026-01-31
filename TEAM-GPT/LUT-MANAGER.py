import yaml

class LUTManager:
    def __init__(self, lut_path):
        self.lut = self.load_lut(lut_path)

    def load_lut(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def query(self, tags):
        # simple intersection match
        matches = {k:v for k,v in self.lut.items() if set(tags) & set(v['tags'])}
        return matches
