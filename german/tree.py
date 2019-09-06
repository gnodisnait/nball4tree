from collections import deque

class Node:
    
    def __init__(self, word, parent=None):
        self.index = 1
        self.word = word
        self.parent = parent
        self.children = []

    def add_child(self, word):
        if self.has_child(word):
            return

        node = Node(word, self)
        node.index = len(self.children)+1
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
        node = self.parent
        while(node is not None):
            indices.append(node.index)
            node = node.parent
        return indices[::-1]

    def get_path_string(self):
        path = get_path()
        return self.word + " "+" ".join(path)

class Tree:
    '''Tree Datastructure for creating word-hypernym trees.
    '''

    def __init__(self):
        self.root = Node('*root*')

    def add_hypernym_path(self, ordered_path):
        '''Adds a hypernym path of a word. 

        :param ordered_path: ordered list of parents of a word/synset, starting from root
        '''
        node = self.root
        for child in ordered_path[1:]:
            if node.has_child(child):
                node = node.get_child(child)
            else:
                n = node.add_child(child)
                node = n
            

    def write_parent_location_code(self, outputfile):
        '''Writes parent locations of all words into a file.

        :param outputfile: file to write into
        '''
    
        # Traverses the tree using depth-first-search (dfs) approach.
        # Writes the location code of each node into the file.
        def dfs(node, visited, file):
            if node is None: # if entire tree was traversed
                return

            path = ' '.join(node.get_path())
            if not path in visited:
                visited.add(path)
                file.write(node.get_path_string()+"\n")
                if not node.is_leaf():
                    dfs(node.children[0], visited, file)
                else:
                    dfs(node.parent, visited, file)
            else:
                unvisited = None
                # checks if there are unvisited neighbor-nodes
                for child in node.children:
                    unv_path = ' '.join(child.get_path())
                    if not unv_path in visited:
                        unvisited = child
                        break

                if unvisited is not None:
                    dfs(unvisited, visited, file)
                else:
                    dfs(node.parent, visited, file)
    
        node = self.root
        visited = {'1'}
        with open(outputfile, 'w') as file:
            dfs(node.children[0], visited, file)


    def write_tree(self, outputfile):
        '''Writes the elements of the tree into a file.

            The structure of the output follows the breadth-first-search approach.
            
            :param outputfile: the file to write into
        '''

        # Traverses the tree using the breadth-first-approach.
        # Writes children of each traversed node.
        def bfs(node, queue, file):
            if node.is_leaf():
                file.write(node.word+"\n")                
            else:
                file.write(" ".join(node.children))
                file.write('\n')
                queue.extend(node.children)

            if len(queue) == 0: # avoid exception once all elements are traversed
                return
            bfs(queue.popleft(), queue, file)    

        node = self.root
        queue = deque()
        with open(outputfile, 'w') as file:
            bfs(node, queue, file)
        