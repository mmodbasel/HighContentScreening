import torch
import copy
import random
import pickle
import numpy as np
from multiprocessing import Process, Queue
from smiles_splitter import split_smiles
from torch.utils.data import Dataset
from torch.utils.data.distributed import Sampler, DistributedSampler
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
torch.manual_seed(42)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


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
	"""
	A dataset that holds SMILES codes, their names, 
	and their fingerprints.
	"""
	def __init__(self, input_file, vocabulary, fprint_file="", most_similar_file="", quiet=False):
		self.vocabulary = vocabulary
		self.fprint_file = fprint_file

		# Read the SMILES file and generate fingerprints
		# for each SMILES if necessary
		self.smiles, self.names, self.fprints = [], [], []
		self.loadFingerprints()		

		if most_similar_file == "":
			self.most_similar = []
			self.precomputed_similarities = False
		else:
			self.most_similar = pickle.load(open(most_similar_file, "rb"))
			self.precomputed_similarities = True


	def loadFingerprints(self, header=False):
		if self.fprint_file == "":
			# Get the number of lines in the file 
			# (only for tracking progress)
			with open(input_file, "r") as f:
				n_lines = sum([1 for line in f])

			with open(input_file, "r") as f:
				for i, line in enumerate(f, 1):
					# Print the progress
					percentage = float(i) / float(n_lines) * 100
					if not quiet:
						print(f"[*] Progress: [{'='*int(percentage/5):<20}] {percentage:>5.1f}%  ", end="\r")

					# Skip headers
					if header:
						continue

					# Get SMILES from line
					split_line = line.strip().split()
					smiles = split_line[0]

					# Calculate and store fingerprint
					fprint = self.calcFprint(smiles)
					if fprint is None:
						continue
					else:
						self.fprints.append(fprint)

					# Store SMILES
					self.smiles.append(smiles)

					# Store name of compound
					try:
						self.names.append(split_line[1])
					except IndexError:
						self.names.append("unknown")
				if not quiet:
					print()

			# Pickle fingerprints for future use
			pickle.dump(
				(self.smiles, self.names, self.fprints), 
				open(f"{input_file.split('.')[0]}_fprints.pk", "wb")
			)
		else:
			self.smiles, self.names, self.fprints = pickle.load(open(self.fprint_file, "rb"))
		return


	def calcFprint(self, smiles):
		"""
		Calculate the Morgan fingerprint from a
		SMILES code.
		"""
		compound = Chem.MolFromSmiles(smiles)
		if compound is None:
			return None

		fprint = AllChem.GetMorganFingerprintAsBitVect(compound, 2, nBits=1024)
		return fprint


	def __len__(self):
		"""
		Return the number of compounds in the dataset.
		"""
		return len(self.smiles)


	def __getitem__(self, i):
		"""
		Get a tuple containing a tokenized compound, 
		its fingerprint, and its index in the dataset.
		"""
		anchor = self.smiles[i]
		anchor_tokens = split_smiles(anchor)
		fprint = self.fprints[i]

		# Pad tokens with <start> and <end> tags
		anchor_out = self.vocabulary.seqToIndex(['<start>']) + \
			   		 self.vocabulary.seqToIndex(anchor_tokens) + \
			   		 self.vocabulary.seqToIndex(['<end>'])

		return (
			torch.tensor(anchor_out).long(), 
			fprint,
			i,
		)


	def getName(self, i):
		"""
		Return the name of a specific compound
		in the dataset.
		"""
		try:
			return self.names[i]
		except IndexError:
			return None


class SimilaritySampler(Sampler):
	"""
	Returns a randomized iter object of indices with
	size 'size'.
	"""
	def __init__(self, size):
		self.size = size


	def __iter__(self):
		"""
		Create the iter objects holding pairs of 
		anchors and possible positives.
		"""
		# Create a shuffled list of indices so that the
		# sampling order is not always the same
		indices = list(range(self.size))
		random.shuffle(indices)
		return iter(indices)


	def __len__(self):
		"""
		Return the number of compounds contained in 
		the sampler.
		"""
		return self.size


class SimilaritySampler2(DistributedSampler):
	"""
	Returns an iter object over a list of indices where
	each randomly picked compound is followed by a positive
	compound (i.e. the positive is within the top N compounds 
	with the highest similarity to the reference).
	"""
	def __init__(self, fprints, most_similar, args, size):
		self.fprints = fprints
		self.most_similar = most_similar
		self.args = args
		self.size = size
		self.precomputed_similarities = False if args.precomputed_similarities == "" else True


	def getSimilar(self):
		"""
		Find an anchor - similars group and return its
		indices.
		"""
		# Get random anchor index and fingerprint
		random_state = np.random.RandomState()
		anchor_index = random_state.choice(len(self.fprints))
		anchor_fprint = self.fprints[anchor_index]

		if not self.precomputed_similarities:
			# Calculate all similarities to the anchor
			similarities = np.array(DataStructs.BulkTanimotoSimilarity(anchor_fprint, self.fprints))

			# Set self-similarity to 0.
			similarities = np.where(similarities == 1., 0., similarities)

			# Find the 10 most similar compounds to the anchor
			similarities_index_pairs = [(similarity, i) for i, similarity in enumerate(similarities)]
			similarities_sorted = sorted(similarities_index_pairs, key=lambda x: x[0], reverse=True)
			top_similar_indices = [pair[1] for pair in similarities_sorted[:10]]
		else:
			top_similar_indices = self.most_similar[anchor_index]

			# Make sure that there are at least 10 similar compounds per anchor
			if len(top_similar_indices) < 10:
				similarities = np.array(DataStructs.BulkTanimotoSimilarity(anchor_fprint, self.fprints))
				similarities = np.where(similarities == 1., 0., similarities)
				similarities_index_pairs = [(similarity, i) for i, similarity in enumerate(similarities)]
				similarities_sorted = sorted(similarities_index_pairs, key=lambda x: x[0], reverse=True)
				top_similar_indices += [pair[1] for pair in similarities_sorted[:10 - len(top_similar_indices)]]

		# Select a random set of 3 compounds similar to the anchor
		random_top_3 = np.random.choice(top_similar_indices, 3, replace=False)
		return anchor_index, random_top_3


	def getSimilars(self, queue, size):
		"""
		Find n (n=size) samples of anchors and similars.
		When finished, put the created list of indices 
		into the multiprocessing queue.
		"""
		indices = []
		while len(indices) < size:
			anchor_index, similar_indices = self.getSimilar()
			indices.append(anchor_index)
			indices.extend(similar_indices)

			# Print the progress as this will take a while
			percentage = float(len(indices)) / float(size) * 100
			if not self.args.quiet:
				print(f"[*] Sampler: [{'='*int(percentage/5):<20}] {percentage:>5.1f}% ", end="\r")

		queue.put(indices)
		return


	def __iter__(self):
		"""
		Create the iter objects holding pairs of 
		anchors and possible positives.
		"""
		# Use multiple processes to get pairs of anchors and positives
		max_size = len(self.fprints) if len(self.fprints) < self.size else self.size
		size_per_process = max_size // self.args.processes
		queue = Queue()
		indices = []
		processes = []

		# Start the processes
		for _ in range(self.args.processes):
			p = Process(target=self.getSimilars, args=(queue, size_per_process))
			p.start()
			processes.append(p)

		for _ in range(len(processes)):
			indices.extend(queue.get())

		for p in processes:
			p.join()

		if not self.args.quiet:
			print("\n[+] Sampling complete")

		return iter(indices)


	def __len__(self):
		"""
		Return the number of compounds contained in 
		the sampler.
		"""
		return len(self.fprints)


def collate_fn(batch):
	"""
	Collate a list of samples where each sample 
	consists of (anchor, fprint, index) to a tuple 
	of (anchors, similarity matrices).
	"""
	# Find the batch size and max length of anchors
	batch_size = len(batch)
	max_length = max([len(item[0]) for item in batch])

	# Define function(s) to calculate similarity
	tanimoto = lambda x, y: np.where(x == 1, y, 0).sum() / (x.sum() + y.sum() - np.where(x == 1, y, 0).sum())
	cosim = lambda x, y: (x * y).sum() / (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()) + 1e-19)   # Not used

	# Calculate similarity matrix for the batch
	all_fprints = np.array([item[1] for item in batch])
	similarity_matrix = np.array([tanimoto(fprint1, fprint2) for fprint1 in all_fprints for fprint2 in all_fprints])
	similarity_matrix = similarity_matrix.reshape(batch_size, batch_size)

	# Initialize anchors tensor
	anchor = torch.zeros(batch_size, max_length)

	# Collate samples
	for i in range(batch_size):
		length_anchor = len(batch[i][0])

		# Determine the amount of padding needed (index of padding = 2)
		padding_anchor = torch.tensor([2] * (max_length - length_anchor))

		# Pad the anchor
		anchor[i, :length_anchor] = batch[i][0] 
		anchor[i, length_anchor:] = padding_anchor

	# Return anchors in shape (tokens, batch size, embedding size)
	anchor = anchor.transpose(0, 1).long()
	similarity_matrix = torch.from_numpy(similarity_matrix).float()
	return (anchor, similarity_matrix)


def random_split(dataset, lengths):
	"""
	Split a dataset into several parts determined
	by the number of elements in the 'lengths' list. 
	The corresponding length in the list defines
	the number of elements per subset.
	"""
	# Shuffle the dataset
	indices = torch.randperm(len(dataset)).tolist()

	# Determine indices of the datasets for subsets
	lengths = [0] + lengths
	lengths = torch.cumsum(torch.tensor(lengths), 0).tolist()
	indices = [indices[lengths[i]:lengths[i + 1]] for i in range(len(lengths) - 1)]

	# Split the dataset
	datasets = []
	for i in range(len(indices)):
		dset = copy.deepcopy(dataset)
		dset.smiles = [dataset.smiles[j] for j in indices[i]]
		dset.names = [dataset.names[j] for j in indices[i]]
		dset.fprints = [dataset.fprints[j] for j in indices[i]]
		if dataset.precomputed_similarities:
			print(f"[*] Getting new indices for dataset {i}")
			smiles_to_index = {smi: idx for idx, smi in enumerate(dset.smiles)}
			self_added = 0
			dset.most_similar = []
			# Iterate over compounds in the new dataset
			for k, j in enumerate(indices[i], 1):
				percentage = float(k / len(indices[i])) * 100.
				print(f"[*] [{'='*int(percentage/5):<20}] {percentage:>5.1f}% ({k} of {len(indices[i])})", end='\r')
				similar = []
				# Iterate over all most similar compounds to the reference
				for s in dataset.most_similar[j]:
					try:
						# Get new index for each of the most similar compounds
						similar.append(smiles_to_index[dataset.smiles[s]])
					except KeyError:
						pass

				# Make sure there are no samples with zero similars
				if len(similar) == 0:
					similar.append(smiles_to_index[dataset.smiles[j]])
					self_added += 1
				dset.most_similar.append(similar)
			print(f"\n[*] Found {self_added} samples with no very similar compounds in dataset")
		datasets.append(dset)
	return datasets
