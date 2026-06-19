import sys
from pathlib import Path

repo = Path("external/PharmacoMatch").resolve()
sys.path.insert(0, str(repo))

from pharmacomatch.virtual_screening.alignment import PharmacophoreAlignment

target_root = sys.argv[1]
aligner = PharmacophoreAlignment(target_root)
aligner.align_preprocessed_ligands_to_query()
print("Alignment time:", getattr(aligner, "alignment_time", None))
