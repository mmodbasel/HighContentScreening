import torch
from torch.utils.data import Dataset
from smiles_splitter import split_smiles


class Vocabulary:
	"""
	Maps each possible 'token' to an index to 
	create a vocabulary encoding all tokens. 
	Special tokens such as the start, end, and
	padding tokens are also assigned an index.
	"""
	def __init__(self, input_file):
		# Define special tokens
		special_elements = ['<start>', '<end>', '<pad>', '<mask>', '<unk>']

		# Create dictionary to map from a token to an index
		self.ele_to_idx = {ele: i for i, ele in enumerate(special_elements)}

		# Generate tokens from a SMILES file
		tokens = self.getTokens(input_file)

		# Add all unique tokens to ele_to_idx dictionary
		i = len(self.ele_to_idx)
		for token in tokens:
			if token not in self.ele_to_idx:
				self.ele_to_idx[token] = i
				i += 1

		# Create dictionary to map from an index to a token
		self.idx_to_ele = {val: key for key, val in self.ele_to_idx.items()}


	def __len__(self):
		"""
		Return the number of tokens in the vocabulary.
		"""
		return len(self.ele_to_idx)


	def getTokens(self, input_file, header=False):
		"""
		Read a SMILES file and split each SMILES into
		individual tokens. Return a list containing all
		unique tokens.
		"""
		unique_components = []
		with open(input_file, "r") as f:
			# Iterate over all SMILES in the file
			for line in f:
				# Skip headers
				if header:
					continue

				# Get and store unique tokens in the SMILES
				smiles = line.strip().split()[0]
				components = split_smiles(smiles)
				for c in components:
					if c not in unique_components:
						unique_components.append(c)
		return unique_components


	def seqToIndex(self, sequence: list, join=False):
		"""
		Convert a sequence (list) to a list of indices 
		corresponding to tokens in the vocabulary.
		If a token in the sequence is unknown, return
		the index to the '<unk>' token.
		"""
		out = [self.ele_to_idx[ele] if ele in self.ele_to_idx else self.ele_to_idx["<unk>"] for ele in sequence]
		return ''.join([str(i) for i in out]) if join else out


	def indexToSeq(self, sequence: list, join=False):
		"""
		Convert a list of indices to a list of tokens.
		If an index in the sequence is unknown, return
		the '<unk>' token.
		"""
		out = [self.idx_to_ele[idx] if idx in self.idx_to_ele else "<unk>" for idx in sequence]
		return ''.join([str(ele) for ele in out]) if join else out


class CustomDataset(Dataset):
	def __init__(self, vocabulary, smiles_file=None, smiles=None, names=None):
		if smiles_file is None and smiles is None:
			print("[-] Please specify either a SMILES file or a list of SMILES")
			exit()
		elif smiles is not None and names is None:
			print("[-] Please provide a list of names for the SMILES")
			exit()

		self.vocabulary = vocabulary
		self.smiles = [] if smiles is None else smiles
		self.names = [] if smiles is None or names is None else names

		if smiles is None:
			with open(smiles_file, "r") as f:
				for line in f:
					split_line = line.strip().split()
					self.smiles.append(split_line[0])
					try:
						self.names.append(split_line[1])
					except IndexError:
						self.names.append("unknown")


	def __len__(self):
		"""
		Return the number of samples in the dataset.
		"""
		return len(self.smiles)


	def __getitem__(self, i):
		"""
		Return a tokenized SMILES string padded 
		with <start> and <end> tokens.
		"""
		smiles = self.smiles[i]
		tokens = split_smiles(smiles)

		# Pad tokens with <start> and <end> tags
		out = self.vocabulary.seqToIndex(['<start>']) + \
			   self.vocabulary.seqToIndex(tokens) + \
			   self.vocabulary.seqToIndex(['<end>'])

		return torch.tensor(out).long()


	def getName(self, i):
		try:
			return self.names[i]
		except IndexError:
			return None


def collate_fn(batch):
	"""
	Collate individual samples to a batch.
	"""
	batch_size = len(batch)
	max_length = max([len(item) for item in batch])

	data = torch.zeros(batch_size, max_length)

	# Make sure each element in the batch is
	# of length max_length by adding padding
	for i in range(batch_size):
		length = len(batch[i])
		padding = torch.tensor([2] * (max_length - length)) # pad_idx = 2
		data[i, :length] = batch[i] 
		data[i, length:] = padding

	# Transpose data from (batch, tokens) to (tokens, batch)
	data = data.transpose(0, 1)
	return data.long()
