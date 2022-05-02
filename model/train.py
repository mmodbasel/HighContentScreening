import argparse, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader
from os import path, mkdir, remove, environ
from dataset import Vocabulary, CustomDataset, SimilaritySampler2, collate_fn, random_split
import net as net
torch.manual_seed(42)


def parse_arguments():
	"""
	Set up the command line argument parser.
	"""
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--jobname', '-j', type=str, default="run", help='jobname of the training (default: run)')
	parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs (default: 10)')
	parser.add_argument('--input', '-i', type=str, default='smiles.smi', help='input smiles file (default: smiles.smi)')
	parser.add_argument('--out_dir', '-o', type=str, default='output/', help='output directory (default: output/)')
	parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size (default: 64)')
	parser.add_argument('--hidden_dims', type=int, default=256, help='number of hidden dimensions (default: 256)')
	parser.add_argument('--n_layers', '-l', type=int, default=4, help='number of layers (default: 4)')
	parser.add_argument('--n_heads', '-n', type=int, default=4, help='number of attention heads (default: 4)')
	parser.add_argument('--scaling', '-s', type=float, default=20., help='scaling factor for similarity loss (default: 20.0)')
	parser.add_argument('--fingerprints', '-f', type=str, default="", help='fingerprints of the input smiles (pickled, default: None)')
	parser.add_argument('--precomputed_similarities', '-r', type=str, default="", help='file containing precomputed similarities (n most similar) (pickled, default: None)')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
	parser.add_argument('--processes', '-p', type=int, default=4, help='number of processes for batch loading (default: 4)')
	parser.add_argument('--quiet', '-q', default=False, action='store_true', help='write output to a log file rather than stdout (default: False)')
	return parser.parse_args()


def log(msg, args, end=''):
	"""
	Log a message. If args.quiet = True, log the 
	message to a file, if args.quiet = False, print
	the message to stdout.
	"""
	if not args.quiet:
		new_line = '\n' if end != '\r' else ''
		print(f"{msg}{new_line}", end=end)
	else:
		if end != '\r':
			log_file = f"{args.jobname}.log"
			with open(log_file, "a") as f:
				f.write(f"{msg}\n")


def main():
	"""
	Create datasets and train the model.
	"""
	# Parse command line arguments
	args = parse_arguments()

	# Log the command line arguments
	print(f"{args}", args)

	# Ensure that the output directory exists
	if not path.isdir(args.out_dir):
		mkdir(args.out_dir)

	# Create vocabulary and dataset
	log(f"[*] Creating vocabulary and dataset", args)
	vocabulary = Vocabulary(args.input)
	vocabulary_out = f"{args.jobname}_vocabulary.pk"
	pickle.dump(vocabulary, open(vocabulary_out, 'wb'))
	log(f"[+] Vocabulary created and written to {vocabulary_out}", args)
	dataset = CustomDataset(
		args.input, 
		vocabulary, 
		fprint_file=args.fingerprints, 
		most_similar_file=args.precomputed_similarities, 
		quiet=args.quiet
	)
	log("[+] Dataset created", args)

	# Define sizes of training and test sets
	train_size = int(len(dataset) * 0.8)
	test_size = len(dataset) - train_size

	# Split the dataset into training and test set
	log("[*] Splitting dataset", args)
	train_set, test_set = random_split(dataset, [train_size, test_size])

	# Create samplers for the training and test sets
	train_sampler = SimilaritySampler2(train_set.fprints, train_set.most_similar, args, int(train_size))
	test_sampler = SimilaritySampler2(test_set.fprints, test_set.most_similar, args, int(test_size))
	
	# Create data loaders for the training and test sets
	train_loader = DataLoader(train_set, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=4, 
		pin_memory=False,
		sampler=train_sampler,
		collate_fn=collate_fn
	)
	test_loader = DataLoader(test_set, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=4, 
		pin_memory=False,
		sampler=test_sampler,
		collate_fn=collate_fn
	)
	log(f"[*] Size training set: {train_size}", args)
	log(f"[*] Size test set: {test_size}", args)

	# Create the model
	pad_idx = vocabulary.seqToIndex(['<pad>'])[0]
	model = net.Transformer(
		args.hidden_dims,
		len(vocabulary),
		pad_idx, 
		args.n_heads, 
		args.n_layers, 
		args.n_layers,
		args.hidden_dims,
		0.1,
		pad_idx,
		args
	)
	
	checkpoint_callback = ModelCheckpoint(
		monitor="validation_loss",
		save_top_k=3,
		filename="validation_loss-{epoch}",
		save_last=True
	)
	trainer = pl.Trainer(
		replace_sampler_ddp=False, 
		accelerator='dp',
		log_every_n_steps=50,
		max_epochs=1000,
		min_epochs=100,
		default_root_dir='output',
		accumulate_grad_batches=4,
		callbacks=[checkpoint_callback]
	)
	trainer.fit(
		model,
		train_dataloaders=train_loader,
		val_dataloaders=test_loader
	)
	return


if __name__ == '__main__':
	environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	environ['CUDA_LAUNCH_BLOCKING'] = '1'
	try:
		main()
	except KeyboardInterrupt:
		print(f"\n[*] Exiting")
		exit()
