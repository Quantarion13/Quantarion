from hgme_core import HGME

hgme = HGME()
# Example canonical hyperedges
edges = [
    {"PQC","ML-KEM","HQC"},
    {"HQC","KYBER","QUORUM16"}
]
for e in edges:
    hgme.add_edge(e)

print("HGME seeded with canonical edges")
