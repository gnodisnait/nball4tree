from util import GermaNetUtil

dir = './'
file_word_embedding = '../data/de.vec'
output_words = 'output-words.txt'
output_tree = 'output-tree.txt'
output_codes = 'output-codes.txt'

util = GermaNetUtil(dir, file_word_embedding)

print("----------- load tree ------------------------")
tree = util.load_tree(output_words)

print("----------- write parent location codes ------")
tree.write_parent_location_code(output_codes)

print("----------- write tree -----------------------")
tree.write_tree(output_tree)

