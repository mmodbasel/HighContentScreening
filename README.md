# HighContentScreening
This repository contains the code to the work on high content similarity-based virtual screening using a distance aware transformer model.

In order to work with this code, you need to install the following dependencies:
* PyTorch
* PyTorch Lightning
* Faiss
* Numpy
* h5py
* rdkit

Alternatively, you can use the `environment.yml` file to create a conda environment:

```bash
conda env create -f environment.yml
conda activate HighContentScreening
```
**Important:** PyTorch will not be installed automatically. You can find the correct version for your CUDA installation [here](https://pytorch.org/get-started/locally/). Also, you have to manually install the correct version of faiss for your system (regular installation: `pip install faiss`, CPU-only version: `pip install faiss-cpu`).

## Dataset
We included a dataset of almost 500,000 SMILES strings in the `dataset` directory. To speed up the training process, we recommend to use pre-computed fingerprints and similarities. These files can be created as shown below:

```bash
# Create fingerprints for the dataset
python create_fingerprints.py dataset.smi

# Compute similarities
python precompute_most_similar.py dataset_fprints.pk 100
```
Where the arguments for `precompute_most_similar.py` are a pickled list containing the fingerprints of the compounds in the dataset and the number of most similar compounds to store for each reference compound.
**Note:** Depending on the size of the dataset and the available resources, these processes may take a very long time.

## Training
You can use the files in the `model` directory to train a model:

```bash
python train.py --hidden_dims 256 --input ../dataset/dataset.smi --jobname HighContentScreening_run1 --lr 0.0001 --out_dir output --scaling 10 --fingerprints ../dataset/dataset_fprints.pk --precomputed_similarities ../dataset/dataset_fprints_similarities_top100.pk
```

**Note:** Since we use pytorch-lightning to train our model, you can track the training progress using tensorboard.

## Similarity Prediction
To use the model in a virtual screening task, the scripts in the `screen` directory can be used. 
First, encode the database you want to screen into latent space. The database needs to be a set of at least one SMILES file in a single directory. The following command can be used to create the encodings:

```bash
python encode.py --checkpoint checkpoint_of_model.pt --vocab vocabulary_of_model.pk --input database/
```
Where `checkpoint_of_model.pt` is a checkpoint file of a model trained in the above step and `vocabulary_of_model.pk` is the vocabulary created during the training of the same model. `database/` is the path to the directory containing all SMILES files of the compounds that need to be encoded. By default, a new directory called `encoded` will be created that holds all encodings. The complete database will be split into several batches containing a fixed number of compounds (default: 10,000,000, can be changed via command line arguments).

Finally, to predict the similarities, the following command can be used:

```bash
python calculate_similarities.py --encodings encoded/ --compounds references/ --checkpoint checkpoint_of_model.pt --vocab vocabulary_of_model.pk
```
Where `encoded/` is the directory containing the encodings, `references/` is a directory containing at least one SMILES file containing compounds to use as reference for the screening, and `--checkpoint` and `--vocab` take the same arguments as for the encoding.
The results will be written to a new directory (default: `most_similar/`). For each reference file, there will be a pickled file containing a dictionary that holds the individual reference compounds as keys and the n most similar compounds to the reference as values (by default n=100000, can be changed as command line argument in `calculate_similarities.py`).

To quickly calculate the Euclidian distance between two compounds, the following command can be used:

```bash
python get_distance.py --reference {smiles1} --target {smiles2} --checkpoint checkpoint_of_model.pt --vocab vocabulary_of_model.pk
```
Where `{smiles1}` and `{smiles2}` are SMILES strings of two compounds to predict the similarity of.