import argparse, pickle, h5py
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import net as net
import numpy as np
from datetime import datetime
from os import path, listdir, mkdir
from dataset import Vocabulary, CustomDataset, collate_fn


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--encodings", "-e", type=str, default="encoded/", help="The directory containing the pickled fingerprints (default: encoded/)")
	parser.add_argument("--compounds", "-c", type=str, default="compounds/", help="The directory containing the smiles files with reference compounds (default: compounds/)")
	parser.add_argument("--outdir", "-o", type=str, default="most_similar/", help="The output directory (default: output/)")
	parser.add_argument('--checkpoint', '-k', type=str, default="checkpoint.pt", help='A stored model checkpoint (default: checkpoint.pt)')
	parser.add_argument('--vocab', '-v', type=str, default="vocabulary.pk", help='A stored vocabulary (default: vocabulary.pk)')
	parser.add_argument('--hidden_dims', type=int, default=256, help='Number of hidden dimensions (default: 256)')
	parser.add_argument('--n_layers', '-l', type=int, default=4, help='Number of layers (default: 4)')
	parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (default: 4)')
	parser.add_argument('--n_similar', type=int, default=100000, help='Number most similar compounds to store (default: 100000)')
	parser.add_argument('--verbose', action='store_true', default=False, help='Print stages of similarity search')
	args = parser.parse_args()
	return args


def load_checkpoint(args):
	"""
	Loads a vocabulary from a pickled file and a
	model from a checkpoint file.
	"""
	# Load vocabulary
	vocabulary = pickle.load(open(args.vocab, "rb"))

	# Load model
	pad_idx = vocabulary.seqToIndex(['<pad>'])[0]
	model = net.Transformer.load_from_checkpoint(
		args.checkpoint, 
		embedding_size=args.hidden_dims, 
		vocab_size=len(vocabulary), 
		src_pad_index=pad_idx, 
		num_heads=args.n_heads, 
		num_encoder_layers=args.n_layers, 
		num_decoder_layers=args.n_layers, 
		forward_expansion=args.hidden_dims, 
		dropout=0.1, 
		device=args.device, 
		pad_idx=pad_idx, 
		args=args
	).to(args.device)
	model.eval()
	return model, vocabulary


def get_reference(args, reference_file, model, vocabulary):
	"""
	Encode the reference SMILES into latent space.
	"""
	# Read SMILES from file
	smiles = []
	names = []
	with open(reference_file, "r") as f:
		for line in f:
			try:
				smi, name = line.strip().split()
			except ValueError:
				continue
			smiles.append(smi)
			names.append(name)

	# Get encodings
	encodings = []
	dataset = CustomDataset(vocabulary, smiles=smiles, names=names)
	for i in range(len(dataset)):
		sample = dataset[i].unsqueeze(0).transpose(0, 1).to(args.device)   # Shape (tokens, 1)
		encoding = model.encode(sample).detach().cpu()   # Shape (tokens, 1, embedding)

		# Get masked mean of encoding
		mask = (model.make_src_mask(sample) == 0).transpose(0, 1).cpu()   # Shape (tokens, 1)
		encoding = (torch.sum(encoding * mask.unsqueeze(2), dim=0) / torch.sum(mask, dim=0).unsqueeze(1)).squeeze(0).numpy()

		encodings.append(encoding)
	return smiles, encodings


def read_data(smiles_file, encoding_file):
	"""
	Read a database of encoded SMILES strings.
	"""
	smiles = pickle.load(open(smiles_file, "rb"))
	dataset = h5py.File(encoding_file, "r")["data"]
	encodings = np.zeros(dataset.shape, dtype='float32')
	dataset.read_direct(encodings)
	return smiles, encodings


def get_similarities(args, reference_file, all_smiles, all_encodings, model, vocabulary, n=1000):
	"""
	Calculate similarities based on L2 norm.
	"""
	# Get reference compounds and embeddings
	smiles_ref, encodings_ref = get_reference(args, reference_file, model, vocabulary)

	# Calculate similarities to reference compounds
	all_most_similar = {}
	query = np.array(encodings_ref)
	for file_idx in range(len(all_smiles)):
		# Print progress
		progress = (file_idx + 1) / len(all_smiles) * 100.
		print(f"[*] Progress: [{'='*int(progress/5):<20}] {progress:>5.1f}%  ", end='\r')

		# Read encoded SMILES strings
		if args.verbose:
			print()
			print(f"[*] Processing file {file_idx + 1} of {len(all_smiles)}")
			print("[*] Reading encodings")
			start = datetime.now()
		smiles, encodings = read_data(all_smiles[file_idx], all_encodings[file_idx])

		# Constructing Faiss index
		if args.verbose:
			print(f"[*] Encodings read in {datetime.now() - start}")
			print("[*] Constructing index")
			start = datetime.now()
		index = faiss.IndexFlatL2(args.hidden_dims)

		# Add encodings to index and search index
		index.add(encodings)
		if args.verbose:
			print(f"[*] Index constructed in {datetime.now() - start}")
			print("[*] Searching index")
			start = datetime.now()
		distances, indices = index.search(query, n)
		if args.verbose:
			print(f"[*] Index searched in {datetime.now() - start}")
			print(f"[*] Most similar to first reference: {smiles[indices[0][0]]} - {distances[0][0]:.3f}")

		# Create (smiles, distance) pairs for all reference compounds
		for i in range(len(encodings_ref)):
			pairs = [(smiles[idx], dist) for idx, dist in zip(indices[i], distances[i])]
			all_most_similar.setdefault(smiles_ref[i], []).extend(pairs)

		# Update most similar to keep only n most similar
		for key, val in all_most_similar.items():
			most_similar = sorted(val, key=lambda x: x[1])[:n]
			all_most_similar[key] = most_similar

		# Clear memory
		del smiles
		del encodings
		del index
		del distances
		del indices

	# Get final set of n most similar
	for key, val in all_most_similar.items():
		most_similar = sorted(val, key=lambda x: x[1])[:n]
		all_most_similar[key] = most_similar
	if not args.verbose:
		print()
	return all_most_similar


def main(args):
	use_cuda = True if torch.cuda.is_available() else False
	device = torch.device('cuda' if use_cuda else 'cpu')
	args.device = device

	# Make sure the output directory exists
	if not path.isdir(args.outdir):
		mkdir(args.outdir)

	# Get smiles and fingerprint files
	all_smiles = sorted([path.join(args.encodings, f) for f in listdir(args.encodings) if "smiles" in f and f[-3:] == ".pk"])
	all_encodings = sorted([path.join(args.encodings, f) for f in listdir(args.encodings) if "encodings" in f and f[-5:] == ".hdf5"])

	# Make sure that there are as many dataset files as fingerprint files
	assert len(all_smiles) == len(all_encodings)

	# Load the model and vocabulary
	model, vocabulary = load_checkpoint(args)
	
	# Get files with reference compounds
	reference_files = sorted([path.join(args.compounds, f) for f in listdir(args.compounds) if f[-4:] == ".smi"])

	# Process all reference files
	for reference_file in reference_files:
		print(f"[*] Processing {reference_file}")
		most_similar = get_similarities(args, reference_file, all_smiles, all_encodings, model, vocabulary, n=args.n_similar)

		# Store most similar
		out_name = reference_file.split('/')[-1].replace(".smi", "-most_similar.pk")
		out_name = path.join(args.outdir, out_name)
		pickle.dump(most_similar, open(out_name, "wb"))
		print(f"[+] Most similar written to {out_name}")
	return


if __name__ == '__main__':
	args = parse_args()
	main(args)

