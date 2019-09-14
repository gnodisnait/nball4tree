from germanet.util import GermaNetUtil

dir = 'data/GermaNet/'
file_word_embedding = 'data/w2v.vec'
output_words = 'data/ws_words.txt'
output_tree = 'data/ws_children.txt'
output_codes = 'data/ws_catcodes.txt'

util = GermaNetUtil(dir, file_word_embedding)

print("----------- loading tree -----------------------")
tree = util.load_tree(output_words)

print("----------- writing parent location codes ------")
tree.write_parent_location_code(output_codes)

print("----------- writing tree -----------------------")
tree.write_tree(output_tree)


words = []
leafs = []
with open(output_tree, 'r') as f:
	for line in f.readlines():
		words.append(line[:-1])
		if len(line[:-1].split()) == 1:
			leafs.append(line[:-1])

codes = []
with open(output_codes, 'r') as f:
	for line in f.readlines():
		codes.append(line.split()[0])

print("tree:")
print("nodes="+str(len(codes)))
print("leafs="+str(len(leafs)))
print("max-depth="+str(tree.depth))