def split_smiles(smiles):
	"""
	Split a SMILES code into individual components.
	"""
	components = []
	double_ele = False
	for i in range(1, len(smiles) + 1):
		if double_ele:
			if smiles[-i].isupper():
				upper = -i + 2 if i > 2 else None
				ele = smiles[-i:upper]
				components.insert(0, ele)
			else:
				ele1 = smiles[-i + 1]
				ele2 = smiles[-i]
				components.insert(0, ele1)
				components.insert(0, ele2)
			double_ele = False
			continue
		else: 
			ele = smiles[-i]

		if not ele.islower():
			components.insert(0, ele)
		elif ele == "c":
			components.insert(0, ele)
		else:
			double_ele = True
	return components


if __name__ == '__main__':
	from sys import argv
	import random

	try:
		input_file = argv[1]
	except IndexError:
		print(f"[-] Please specify an input smiles file")

	smiles = []
	with open(input_file, "r") as f:
		smiles = [line.strip().split()[0] for line in f]

	for s in smiles:
		components = split_smiles(s)
		if random.random() > 0.99:
			print(s)
			print(components)