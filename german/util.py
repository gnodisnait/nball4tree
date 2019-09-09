from pygermanet import load_germanet
from bs4 import BeautifulSoup
from tree import Tree
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
		self.__extract_words(germanet, outputfile)
		words = []
		with open(outputfile, 'r') as file:
			words = file.readlines()
		
		# step 2: fill tree with hypernym paths
		
		tree = Tree()
		for word in words:
			synset = germanet.synset(word[:-1]) 
			if synset is None:
				continue
			paths = synset.hypernym_paths

			# checks if synset has multiple paths
			if len(paths) > 0 and isinstance(paths[0], list):
				for path in paths:
					tree.add_hypernym_path(path)
			else:
				tree.add_hypernym_path(paths)

		return tree

	def __get_synset_name(self, synset):
		return synset.__str__()[7:-1]

	def __extract_words(self, germanet, outputFile):
		'''Extracts words from xml files. 
		Only those contained in the word-vector embedding are kept.

		:param outputFile: destination file 
		'''

		# skips everything if file was already written into
		if os.path.isfile(outputFile):
			return

		dir = self.__source
		wordVecFile = self.__w2vec_file

		# gets all xml files
		def getFiles(dir):
			prefixes = ['adj', 'verben', 'nomen']
			files = [dir+file for file in os.listdir(dir) if file.split('.')[0] in prefixes]
			return files

		# extracts all existing words from the word-embedding file
		def get_vector_words(file):
			words = []
			with open(file, 'r') as f_in:
				lines = f_in.readlines()
				for line in lines:
					words.append(line.split()[0])
			return set(words)

		def get_words(files):
			words = []
			for file in files:
				with open(file, 'r') as f:
					xml = BeautifulSoup(f.read(), 'html.parser')
					for child in xml.find_all('synset'):
						for lexUnit in child.find_all('lexunit'):
							word = lexUnit.orthform.text
							synsets = germanet.synsets(word)
							for synset in synsets:
								name = self.__get_synset_name(synset)
								# avoids multiple word-compositions
								if len(name.split()) == 1:
									words.append(name)
			return set(words)		

		files = getFiles(dir)
		
		words = get_words(files)
		# step 1: get all words from word-embedding
		embedded_words = get_vector_words(wordVecFile)
		
		retained_words = words.intersection(embedded_words)

		with open(outputFile, 'w') as f:
			f.write("\n".join(retained_words))