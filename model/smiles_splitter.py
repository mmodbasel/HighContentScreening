def split_smiles(smiles):
	"""
	Split a SMILES code into individual components.
	"""
	components = []
	double_ele = False
	# Iterate over SMILES string from back to front
	for i in range(1, len(smiles) + 1):
		# Handle cases where an element consists of two letters
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
			previous = smiles[-(i + 1)] if i < len(smiles) else smiles[-i]

		# Find out if the element consists of two letters
		# If it does not --> store the token
		if not ele.islower():
			components.insert(0, ele)
		elif ele == "c":
			components.insert(0, ele)
		elif ele == "n" and previous not in  ["M", "Z", "I", "S"]:
			components.insert(0, ele)
		else:
			double_ele = True
	return components


if __name__ == '__main__':
	"""
	Print all unique components contained in 
	a SMILES file.
	"""
	from sys import argv
	import random

	try:
		input_file = argv[1]
	except IndexError:
		print(f"[-] Please specify an input smiles file")

	smiles = []
	with open(input_file, "r") as f:
		smiles = [line.strip().split()[0] for line in f]

	unique_components = []
	for s in smiles:
		components = split_smiles(s)
		for component in components:
			if component not in unique_components:
				unique_components.append(component)

		# Periodically print a SMILES string and 
		# its components
		if random.random() > 0.9999:
			print(s)
			print(components)
	print(unique_components)