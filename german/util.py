from pygermanet import load_germanet
from bs4 import BeautifulSoup
from .tree import Tree
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

		# step 1: extract words from GermaNet
		self.__extract_words(outputfile)
		words = []
		with open(outputfile, 'r') as file:
			words = file.readlines()

		# step 2: fill tree with hypernym paths
		germanet = load_germanet()
		tree = Tree()
		for word in words:
			paths = germanet.synset(word).hypernym_paths
			for path in paths:
				tree.add_hypernym_path(path)

		return tree

	def __extract_words(self, outputFile):
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
			return words

		# step 1: get all words from word-embedding
		embeddedWords = get_vector_words(wordVecFile)

		# step 2: remove all words in germanet that are not present in the word-embeddings
		# 		  and write the remaining words into a file
		files = getFiles(dir)
		with open(outputFile, 'w') as out:
			for file in files:
				with open(file, 'r') as f:
					xml = BeautifulSoup(f.read(), 'html.parser')
					for child in xml.find_all('synset'):
						for lexUnit in child.find_all('lexunit'):
							word = lexUnit.orthform.text
							if word in embeddedWords:
								out.write(lexUnit.orthform.text)
								out.write('\n')		