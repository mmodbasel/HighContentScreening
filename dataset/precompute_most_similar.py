import pickle
import numpy as np
from sys import argv
from rdkit import DataStructs, RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
	data_file, n_most_similar = argv[1], int(argv[2])
except IndexError:
	print("[-] Please specify 1) a pickled file containing a list of fingerprints and 2) the number of most similar compounds to store")
	exit()
except ValueError:
	print(f"[-] \"{n_most_similar}\" is an invalid option for the number of most similar compounds")
	exit()

_, _, fprints = pickle.load(open(data_file, "rb"))
most_similar = []
for fprint in fprints:
	similarities = np.array(DataStructs.BulkTanimotoSimilarity(fprint, fprints))
	similarities = np.where(similarities == 1., 0., similarities)
	similarity_index_pairs = [(similarity, i) for i, similarity in enumerate(similarities)]
	similarities_sorted = sorted(similarity_index_pairs, key=lambda x: x[0], reverse=True)
	top_similar_indices = [pair[1] for pair in similarities_sorted[:n_most_similar]]
	most_similar.append(top_similar_indices)

out_name = data_file.replace(".pk", f"_similarities_top{n_most_similar}.pk")
pickle.dump(most_similar, open(out_name, "wb"))
print(f"[+] Data written to {out_name}")
