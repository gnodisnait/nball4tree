words = []
leafs = []
with open('output-tree.txt', 'r') as f:
	for line in f.readlines():
		words.append(line[:-1])
		if len(line[:-1].split()) == 1:
			leafs.append(line[:-1])

codes = []
with open('output-codes.txt', 'r') as f:
	for line in f.readlines():
		codes.append(line.split()[0])

print("tree:")
print("----------")
print("nodes="+str(len(codes)))
print("leafs="+str(len(leafs)))

