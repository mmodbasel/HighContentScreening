import argparse, math, pickle, h5py
import net
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset as dataset_lightning
from torch.utils.data import DataLoader
from dataset import Vocabulary, CustomDataset, collate_fn
from os import path, listdir, mkdir


def parse_arguments():
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--checkpoint', '-c', type=str, default="checkpoint.pt", help='A stored model checkpoint (default: checkpoint.pt)')
	parser.add_argument('--vocab', '-v', type=str, default="vocabulary.pk", help='A stored vocabulary (default: vocabulary.pk)')
	parser.add_argument('--batch_size', '-b', type=int, default=128, help='The batch size used for the model (default: 128)')
	parser.add_argument('--input', '-i', type=str, default='dataset/', help='Directory containing the input smiles file (default: dataset/)')
	parser.add_argument('--outdir', '-o', type=str, default='encoded/', help='The output directory (default: encoded/)')
	parser.add_argument('--hidden_dims', type=int, default=256, help='Number of hidden dimensions (default: 256)')
	parser.add_argument('--n_layers', '-l', type=int, default=4, help='Number of layers (default: 4)')
	parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (default: 4)')
	parser.add_argument('--n_per_batch', '-n', type=int, default=10000000, help='Number of compounds to write per batch (default: 10000000)')
	return parser.parse_args()


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


def smiles_generator(all_files, batch_size=10000):
	"""
	Generator object that splits SMILES strings
	in a set of smi files into batches of fixed size.
	"""
	smiles, names = [], []
	# Iterate over all files
	for smiles_file in all_files:
		with open(smiles_file, "r") as f:
			for line in f:
				try:
					# Extract the SMILES and name of the compound
					smi, name = line.strip().split()
				except ValueError as e:
					print(f"[-] ValueError in line:\n{line}\n({e})")
					continue
				smiles.append(smi)
				names.append(name)

				# Once the batch has the specified size,
				# yield the data
				if len(smiles) >= batch_size:
					yield smiles, name
					smiles, names = [], []
	yield smiles, names


def process_batch(args, processing_batch, generator, model, vocabulary):
	# Check if outfiles already exist
	encodings_out = path.join(args.outdir, f"encodings{processing_batch}.hdf5")
	smiles_out = path.join(args.outdir, f"smiles{processing_batch}.pk")
	if path.exists(encodings_out) and path.exists(smiles_out):
		smiles = pickle.load(open(smiles_out, "rb"))
		_ = next(generator)
		return len(smiles), generator, False

	# Get SMILES and names in batch
	print(f"[*] Processing batch {processing_batch}")
	try:
		smiles, names = next(generator)
	except StopIteration:
		print("[+] All batches finished")
		return 0, None, True

	# Create dataset
	dataset = CustomDataset(vocabulary, smiles=smiles, names=names)
	n_compounds = len(dataset)

	# Create data loader (for parallelization)
	data_loader = DataLoader(
		dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		pin_memory=True, 
		collate_fn=collate_fn,
		num_workers=3
	)

	# Encode all files in batch
	encodings = []
	all_smiles = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			# Get start and end indices of batch
			start_idx = batch_idx * args.batch_size
			end_idx = (batch_idx + 1) * args.batch_size
			end_idx = end_idx if end_idx <= len(dataset) else len(dataset)

			# Encode SMILES
			batch = batch.to(args.device)
			encoding = model.encode(batch).detach().cpu()

			# Get masked mean of encoding
			mask = (model.make_src_mask(batch) == 0).transpose(0, 1).cpu()   # Mask in shape (tokens, batch)
			# Encoding in shape (tokens, batch, embedding)
			# Get masked mean over tokens --> shape (batch, embedding)
			encoding = (torch.sum(encoding * mask.unsqueeze(2), dim=0) / torch.sum(mask, dim=0).unsqueeze(1)).numpy()

			# Store SMILES and encodings
			all_smiles.extend(dataset.smiles[start_idx:end_idx])
			encodings.extend([encoding[i] for i in range(encoding.shape[0])])

	# Write SMILES and encodings to file
	with h5py.File(encodings_out, "w") as f:
		f.create_dataset("data", data=encodings)
	print(f"[+] Encodings of batch {processing_batch} written to {encodings_out}")
	pickle.dump(all_smiles, open(smiles_out, "wb"))
	print(f"[+] SMILES of batch {processing_batch} written to {smiles_out}")
	return n_compounds, generator, False


def main(args, batch_size=10000000):
	use_cuda = True if torch.cuda.is_available() else False
	device = torch.device('cuda' if use_cuda else 'cpu')
	args.device = device

	# Make sure the output directory exists
	if not path.isdir(args.outdir):
		mkdir(args.outdir)

	# Load the model and vocabulary
	print(f"[*] Loading vocabulary and model")
	model, vocabulary = load_checkpoint(args)
	
	# Find all smiles files in dataset
	all_files = sorted([path.join(args.input, f) for f in listdir(args.input) if f[-4:] == ".smi"])

	# Create encodings for all input files
	n_compounds = 0
	batch = 0
	generator = smiles_generator(all_files, batch_size=batch_size)
	while True:
		batch += 1
		compounds, generator, stop = process_batch(args, batch, generator, model, vocabulary)
		n_compounds += compounds
		if stop:
			break

	print(f"[+] Processed all files ({n_compounds} compounds in total)")


if __name__ == '__main__':
	args = parse_arguments()
	try:
		main(args, batch_size=args.n_per_batch)
	except KeyboardInterrupt:
		print(f"\n[*] Exiting")
		exit()
