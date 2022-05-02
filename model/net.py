import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from dataset import SimilaritySampler2, collate_fn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
torch.manual_seed(42)


class PositionalEncoding(LightningModule):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		"""
		Implements positional encoding according to Vaswani et al (2017):
		https://doi.org/10.48550/arXiv.1706.03762)
		"""
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

			
class Transformer(LightningModule):
	def __init__(
		self,
		embedding_size,
		vocab_size,
		src_pad_index,
		num_heads,
		num_encoder_layers,
		num_decoder_layers,
		forward_expansion,
		dropout,
		pad_idx,
		args
	):
		super(Transformer, self).__init__()
		self.args = args
		self.pad_idx = pad_idx

		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.position_embedding = PositionalEncoding(embedding_size, dropout)
		# Use PyTorch's transformer implementation
		self.transformer = nn.Transformer(
			embedding_size, 
			num_heads,
			num_encoder_layers, 
			num_decoder_layers, 
			forward_expansion,
			dropout
		)

		self.fc_out = nn.Linear(embedding_size, vocab_size)
		self.src_pad_idx = src_pad_index


	def make_src_mask(self, src):
		# src has shape (N_tokens, batch size)
		src_mask = src.transpose(0, 1) == self.src_pad_idx
		return src_mask


	def forward(self, src, trg):
		# Do positional encoding of data
		embed_src = self.position_embedding(self.embedding(src))
		embed_trg = self.position_embedding(self.embedding(trg))

		# Create src padding mask
		src_padding_mask = self.make_src_mask(src)

		# Compute encoding
		encoded = self.transformer.encoder(
			embed_src,
			src_key_padding_mask=src_padding_mask
		)
		return encoded


	def pairwise_distances(self, embeddings, squared=False):
		"""
		Calculate pairwise distances between all embedded
		samples in a batch. If squared = True, the squared
		distances are returned without taking the square
		root. 
		"""
		# Shape of embeddings: (batch, features)
		# Shape of dot product: (batch, batch)
		dot_product = torch.einsum('af,bf->ab', (embeddings, embeddings))

		# Get squared L2 norm of embeddings (diagonal of the dot product gives ||a|| & ||b||) 
		# Shape square norm: (batch,)
		square_norm = torch.diagonal(dot_product, dim1=0, dim2=1)

		# Compute pairwise distance matrix
		# ||a - b||^2 = ||a||^2 - 2ab + ||b||^2
		# Shape distances: (batch, batch)
		distances = square_norm.unsqueeze(0) - 2. * dot_product + square_norm.unsqueeze(1)

		# Ensure that there are no negative distances
		distances = torch.where(distances > torch.zeros_like(distances), distances, torch.zeros_like(distances))

		# Get the square root of the distances
		if not squared:
			# Make sure no values are exactly 0
			mask = torch.where(distances == 0., torch.ones_like(distances), torch.zeros_like(distances))
			distances = distances + mask * 1e-16

			# Take the square root of the distances
			distances = torch.sqrt(distances)

			# Correct for the added epsilon
			distances = distances * (1.0 - mask)

		return distances


	def similarity_loss_fn(self, encoded, similarity_matrix, args, squared=False):
		"""
		Calculate and return the similarity loss of a batch.
		"""
		# Take the mean over all tokens in the embedding
		# (tokens, batch, features) -> (batch, features)
		embeddings = encoded.mean(dim=0)

		# Get the (embedding) distances between all compounds in the batch
		distance_matrix = self.pairwise_distances(embeddings, squared=squared)

		# Calculate similarity loss
		similarity_loss = torch.abs((1 - similarity_matrix) * args.scaling - distance_matrix)
		return similarity_loss.mean()


	def training_step(self, batch, batch_idx):
		anchor, similarity_matrix = batch # Shape of anchor is (N_tokens, batch size)
		src, trg = anchor, anchor[:-1]
		trg_seq_length, N = trg.shape
		
		embed_src = self.position_embedding(self.embedding(src))
		embed_trg = self.position_embedding(self.embedding(trg))

		# Create masks
		src_padding_mask = self.make_src_mask(src)
		trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).type_as(trg).float()

		# Encode data
		encoded = self.transformer.encoder(
			embed_src,
			src_key_padding_mask=src_padding_mask
		)
		# Decode data
		decoded = self.transformer.decoder(
			embed_trg,
			encoded,
			tgt_mask=trg_mask
		)
		
		# Calculate similarity loss
		similarity_loss = self.similarity_loss_fn(encoded, similarity_matrix, self.args)

		# Reshape the data and calculate reconstruction loss
		encoded = encoded.reshape(-1, encoded.shape[-1])
		decoded = decoded.reshape(-1, decoded.shape[-1])
		target = anchor[1:].reshape(-1) 
		reconstruction_loss = F.cross_entropy(
			decoded, 
			target, 
			ignore_index=self.pad_idx
		)

		# Log similarity and reconstruction loss
		self.log("train_sim_loss", similarity_loss, sync_dist=True)
		self.log("train_rec_loss", reconstruction_loss, sync_dist=True)

		# The loss consists of the sum of triplet and reconstruction loss
		loss = similarity_loss + reconstruction_loss
		self.log("train_loss", loss, sync_dist=True)
		return loss


	def validation_step(self, batch, batch_idx):
		anchor, similarity_matrix = batch # Shape of anchor is (N_tokens, batch size)
		src, trg = anchor, anchor[:-1]
		trg_seq_length, N = trg.shape

		embed_src = self.position_embedding(self.embedding(src))
		embed_trg = self.position_embedding(self.embedding(trg))

		# Create masks
		src_padding_mask = self.make_src_mask(src)
		trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).type_as(trg).float()

		# Encode data
		encoded = self.transformer.encoder(
			embed_src,
			src_key_padding_mask=src_padding_mask
		)
		# Decode data
		decoded = self.transformer.decoder(
			embed_trg,
			encoded,
			tgt_mask=trg_mask
		)

		# Calculate similarity loss
		similarity_loss = self.similarity_loss_fn(encoded, similarity_matrix, self.args)

		# Reshape the data and calculate reconstruction loss
		encoded = encoded.reshape(-1, encoded.shape[-1])
		decoded = decoded.reshape(-1, decoded.shape[-1])
		target = anchor[1:].reshape(-1) 
		reconstruction_loss = F.cross_entropy(
			decoded, 
			target, 
			ignore_index=self.pad_idx
		)

		# Log similarity and reconstruction loss
		self.log("val_sim_loss", similarity_loss, sync_dist=True)
		self.log("val_rec_loss", reconstruction_loss, sync_dist=True)

		# The loss consists of the sum of triplet and reconstruction loss
		loss = similarity_loss + reconstruction_loss

		self.log("validation_loss", loss, sync_dist=True)
		return loss


	def encode(self, src):
		"""
		Encode a tokenized SMILES string.
		"""
		with torch.no_grad():
			embed_src = self.position_embedding(self.embedding(src))
			mask = self.make_src_mask(src)
			encoded = self.transformer.encoder(embed_src, src_key_padding_mask=mask)
		return encoded


	def configure_optimizers(self):
		"""
		Define the optimizer to use during training
		"""
		optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
		return optimizer
