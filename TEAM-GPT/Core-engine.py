# hgme_core.py
class HGME:
    def __init__(self):
        self.nodes = set()
        self.hyperedges = []

    def add_edge(self, edge: set):
        self.hyperedges.append(edge)
        self.nodes |= edge

    def retrieve(self, active_tags: set):
        scores = {n: 0 for n in self.nodes}
        for e in self.hyperedges:
            if e & active_tags:
                for n in e:
                    scores[n] += 1 / len(e)
        # sort descending
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

# Fusion example
def phi_fusion(lut_output, hg_scores, phi=1.9102):
    if lut_output:
        return lut_output
    return {k: v*phi for k, v in hg_scores.items()}
