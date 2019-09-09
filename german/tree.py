import os 

class Node:
    
    def __init__(self, word, index=1, parent=None):
        self.index = index
        self.word = word
        self.parent = parent
        self.children = []

    def add_child(self, word):
        if self.has_child(word):
            return

        index = len(self.children)+1
        node = Node(word, index, self)
        self.children.append(node)
        return node

    def has_child(self, word):
        return self.get_child(word) is not None

    def get_child(self, word):
        for child in self.children:
            if child.word == word:
                return child

    def is_leaf(self):
        return not self.children
    
    def get_path(self):
        indices = []
        node = self
        while(node is not None):
            indices.append(str(node.index))
            node = node.parent
        return indices[::-1]

    def get_path_string(self, depth=None):
        path = self.get_path()
        
        if depth is not None:
            length = depth - len(path)
            path += ['0']*length
        
        return self.word + " "+" ".join(path)

class Tree:
    '''Tree Datastructure for creating word-hypernym trees.
    '''

    def __init__(self):
        self.root = Node('*root*')
        self.depth = 1

    def add_hypernym_path(self, ordered_path):
        '''Adds a hypernym path of a word. 

        :param ordered_path: ordered list of parents of a word/synset, starting from root
        '''
        node = self.root
        for synset in ordered_path[1:]:
            child = synset.__str__()[7:-1]
            
            #avoids nodes with multiple word-compositions
            if len(child.split()) > 1:
                return
            if node.has_child(child):
                node = node.get_child(child)
            else:
                n = node.add_child(child)
                node = n

        
            

    def write_parent_location_code(self, outputfile):
        '''Writes parent locations of all words into a file.

        :param outputfile: file to write into
        '''

        if os.path.isfile(outputfile):
            return

        def traverse(node, visited, file):
            if node.is_leaf():
                return
            for child in node.children:
                code = node.get_path_string(self.depth)
                if not code in visited:
                    file.write(code+"\n")
                    visited.add(code)
                traverse(child, visited, file)
    

        node = self.root
        visited = {'1'}
        with open(outputfile, 'w') as file:
            traverse(node, visited, file)


    def write_tree(self, outputfile):
        '''Writes the elements of the tree into a file.

            The structure of the output follows the breadth-first-search approach.
            
            :param outputfile: the file to write into
        '''  

        def traverse(node, visited, file):
            code = node.get_path_string()
            if node.is_leaf():
                if not code in visited:
                    file.write(node.word+"\n")
                    visited.add(code)
            else:
                if not code in visited:
                    file.write(node.word +" "+" ".join(map(lambda n: n.word, node.children)))
                    file.write('\n')
                    visited.add(code)
                for child in node.children:
                    traverse(child, visited, file)


        node = self.root
        visited = {'1'}
        with open(outputfile, 'w') as file:
            traverse(node, visited, file)
        