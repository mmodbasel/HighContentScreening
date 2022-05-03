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
	parser.add_argument("--reference", "-r", type=str, help="The reference SMILES")
	parser.add_argument("--target", "-t", type=str, help="The target SMILES")
	parser.add_argument('--checkpoint', '-k', type=str, default="checkpoint.pt", help='A stored model checkpoint (default: checkpoint.pt)')
	parser.add_argument('--vocab', '-v', type=str, default="vocabulary.pk", help='A stored vocabulary (default: vocabulary.pk)')
	parser.add_argument('--hidden_dims', type=int, default=256, help='Number of hidden dimensions (default: 256)')
	parser.add_argument('--n_layers', '-l', type=int, default=4, help='Number of layers (default: 4)')
	parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads (default: 4)')
	args = parser.parse_args()
	return args


def load_checkpoint(args):
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


def embed(args, model, vocabulary, smiles):
	# Get encodings
	dataset = CustomDataset(vocabulary, smiles=[smiles], names=[smiles])
	encoding = model.encode(dataset[0].unsqueeze(0).to(args.device)).squeeze(0).mean(dim=0).detach().cpu().numpy()
	return encoding


def get_distance(args, reference, target, n=1):
	index = faiss.IndexFlatL2(args.hidden_dims)
	index.add(np.array([target]))
	distances, indices = index.search(np.array([reference]), n)
	return distances.tolist()[0][0]


def main(args):
	use_cuda = True if torch.cuda.is_available() else False
	device = torch.device('cuda' if use_cuda else 'cpu')
	args.device = device

	# Load the model and vocabulary
	model, vocabulary = load_checkpoint(args)

	# Embed reference and target
	reference = embed(args, model, vocabulary, args.reference)
	target = embed(args, model, vocabulary, args.target)

	# Get distance
	distance = get_distance(args, reference, target)
	if distance is not None:
		print(f"[+] Distance to target: {distance:.3f}")
	return


if __name__ == '__main__':
	args = parse_args()
	main(args)