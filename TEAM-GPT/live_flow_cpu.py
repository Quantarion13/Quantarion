from pathlib import Path
from hgme import HGME  # HyperGraph Memory Engine
from lut import LUTCache  # Conceptual LUT
from polyglot import Translator

# Load memory
lut = LUTCache("./memory/lut/")
hgme = HGME()
hgme.load("./memory/hgme/")

# Input / Output
input_dir = Path("input")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Languages from config
languages = ["en","fr","ru","es","de","zh","ja","pt","it"]

for lang in languages:
    file_path = input_dir / f"{lang}.txt"
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")

        # Step 1: LUT check (deterministic reuse)
        lut_result = lut.lookup(text, lang)
        if lut_result:
            result = lut_result
        else:
            # Step 2: Hypergraph fallback
            hg_scores = hgme.retrieve([lang, *text.split()])

            # Step 3: φ⁴³ Fusion
            phi = 22.93606797749979
            result = {k: v*phi for k,v in hg_scores.items()}

        # Step 4: Save CPU-processed output
        (output_dir / f"{lang}_out.txt").write_text(str(result), encoding="utf-8")
