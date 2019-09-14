from pygermanet import load_germanet
from xml.sax import make_parser, handler
from germanet.tree import Tree
import os


class GermaNetUtil:
	'''A utility class for creating a tree structure out of german hypernyms.'''

	__source = ''
	__w2vec_file = ''
	

	def __init__(self, sourcefolder, wordvec_file):
		'''
		:param sourcefolder: folder containing all xml files from GermaNet
		:param wordvec_file: file containing german word-embeddings
		'''

		self.__source = sourcefolder
		self.__w2vec_file = wordvec_file

	def load_tree(self, outputfile):
		'''Creates a tree and fills it with words and hypernyms.

		:param outputfile: the outputfile to be created containing all words
		:return: complete tree
		'''

		germanet = load_germanet()

		# step 1: extract words from GermaNet
		print("extracting words..")
		words, embedded_words = self.__extract_words(germanet, outputfile)
				
		# step 2: fill tree with hypernym paths
		count = 0
		skipped = 0
		mult_paths = 0
		tree = Tree()
		for word in words:
			synset = germanet.synset(word) 
			if synset is None:
				skipped += 1
				continue
			paths = synset.hypernym_paths

			if len(paths) == 0:
				skipped += 1
				return
			elif len(paths) > 0 and isinstance(paths[0], list): # checks if synset has multiple paths
				mult_paths += 1
				for path in paths:
					count +=1
					tree.add_hypernym_path(path, embedded_words)
			else:
				count+=1
				tree.add_hypernym_path(paths, embedded_words)

		print("number of words added = "+str(len(tree.words)))
		print("number of paths = "+str(count))
		print("number of synsets with multiple paths = "+str(mult_paths))
		print("skipped = "+str(skipped))

		return tree

	def __get_synset_name(self, synset):
		return synset.__str__()[7:-1]

	def __extract_words(self, germanet, outputFile):
		'''Extracts words from xml files. 
		Only those contained in the word-vector embedding are kept.

		:param outputFile: destination file 
		'''

		dir = self.__source
		wordVecFile = self.__w2vec_file

		# extracts all existing words from the word-embedding file
		def get_vector_words(file):
			words = []
			with open(file, 'r') as f_in:
				for line in f_in:
					words.append(line.split()[0])
			return set(words)
	

		def get_words(dir):
			prefixes = ['adj', 'verben', 'nomen']
			files = [dir+file for file in os.listdir(dir) if file.split('.')[0] in prefixes]

			class ContentH(handler.ContentHandler):
				def __init__(self):
					self.words = []
					self.current_content = ""

				def startElement(self, name, attrs):
					self.current_content = ""

				def characters(self, content):
					if len(content.strip().split()) == 1:
						self.current_content += content.strip()

				def endElement(self, name):
					if name.lower() == "orthform" and self.current_content:
						self.words.append(self.current_content)
						
			words = []
			for file in files:
				parser = make_parser()
				content_handler = ContentH()
				parser.setContentHandler(content_handler)
				parser.parse(file)
				words += content_handler.words
			return set(words)



		# step 1: get all words from word-embedding
		embedded_words = get_vector_words(wordVecFile)
		print("number of word embeddings = "+str(len(embedded_words)))

		# skips step 2 and 3
		if os.path.isfile(outputFile):
			synsets = set()
			with open(outputFile, 'r') as file:
				for line in file:
					synsets.add(line[:-1])
				print("number of words in word embeddings and GermaNet = "+str(len(synsets)))
			return synsets, embedded_words

		# step 2: read words from WordNet
		words = get_words(dir)
		print("number of words in GermaNet = "+str(len(words)))
		# step 3: discard words that are not present in word-embedding
		retained_words = words.intersection(embedded_words)
		print("number of same words in word embeddings and GermaNet = "+str(len(retained_words)))

		synsets = []
		with open(outputFile, 'w') as f:
			for word in retained_words:
				lst = [self.__get_synset_name(ele) for ele in germanet.synsets(word)]
				synsets += lst
				f.write("\n".join(lst))
				f.write('\n')

		return set(synsets), embedded_words

		
