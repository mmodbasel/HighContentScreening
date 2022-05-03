import pickle
from sys import argv
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
	data_file = argv[1]
except IndexError:
	print("[-] Please specify a file containing one or more SMILES strings")
	exit()

all_smiles, names, fprints = [], [], []
with open(data_file, "r") as f:
	for line in f:
		smiles = line.strip().split()[0]
		try:
			name = line.strip().split()[1]
		except Exception:
			name = "unknown"

		compound = Chem.MolFromSmiles(smiles)
		if compound is None:
			continue

		fprint = AllChem.GetMorganFingerprintAsBitVect(compound, 2, nBits=1024)
		all_smiles.append(smiles)
		names.append(name)
		fprints.append(fprint)

out_name = data_file.replace(".smi", f"_fprints.pk")
pickle.dump((all_smiles, names, fprints), open(out_name, "wb"))
print(f"[+] Fingerprints written to {out_name}")
