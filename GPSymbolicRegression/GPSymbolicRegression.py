import numpy as np
from copy import deepcopy
import random

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self):
        if self.value in ['+', '-', '*', '/']:
            return f"({str(self.left)} {self.value} {str(self.right)})"
        return str(self.value)

class GeneticSymbolicRegressor:
    def __init__(self, 
                 population_size=31,
                 generations=20,
                 tournament_size=27,
                 mutation_rate=0.1,
                 max_depth=5):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        self.operators = ['+', '-', '*', '/']
        self.best_solution = None
        self.best_loss = float('inf')
        self.loss_history =   []
        
    def generate_random_tree(self, depth=0):
        if depth >= self.max_depth or (random.random() < 0.3):
            return Node(random.uniform(-10, 10) if random.random() < 0.5 else 'x')
        
        node = Node(random.choice(self.operators))
        node.left = self.generate_random_tree(depth + 1)
        node.right = self.generate_random_tree(depth + 1)
        return node

    def evaluate_tree(self, node, x):
        if node.value == 'x':
            return x
        if isinstance(node.value, (int, float)):
            return node.value
        
        left_val = self.evaluate_tree(node.left, x)
        right_val = self.evaluate_tree(node.right, x)
        
        try:
            if node.value == '+':
                return left_val + right_val
            elif node.value == '-':
                return left_val - right_val
            elif node.value == '*':
                return left_val * right_val
            elif node.value == '/':
                return left_val / right_val if right_val != 0 else 1e10
        except:
            return 1e30 

    def loss(self, tree, X, y):
        try:
            predictions = np.array([self.evaluate_tree(tree, x) for x in X])
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return 1e30
            discount = 1.05
            mse = np.mean(np.abs(predictions - y)**2)
            nodes = self.get_nodes(tree)
            return mse
        except:
            return 1e30

    def get_parent(self, root, target):
        if not root:
            return None
        u1, u2, u = id(root.left), id(root.right), id(target)
        if u1 == u or u2 == u:
            return root
            
        left_result = self.get_parent(root.left, target)
        if left_result:
            return left_result
            
        return self.get_parent(root.right, target)

    def get_nodes(self, node, nodes=None, depth=0):
        if nodes is None:
            nodes = []
        if node:
            nodes.append((node, depth))
            self.get_nodes(node.left, nodes, depth + 1)
            self.get_nodes(node.right, nodes, depth + 1)
        return nodes

    def crossover(self, parent1, parent2):
        new_parent1 = deepcopy(parent1)
        new_parent2 = deepcopy(parent2)
        #print(new_parent1)
        nodes1 = self.get_nodes(new_parent1)
        nodes2 = self.get_nodes(new_parent2)
        
        if len(nodes1) < 2 or len(nodes2) < 2:
            return new_parent1
        
        node1 = random.choice(nodes1[1:])[0]
        node2 = random.choice(nodes2[1:])[0]
        
        parent_node1 = self.get_parent(new_parent1, node1)
        parent_node2 = self.get_parent(new_parent2, node2)
        
        if not parent_node1 or not parent_node2:
            return new_parent1
        
        if parent_node1.left == node1:
            parent_node1.left = deepcopy(node2)
        else:
            parent_node1.right = deepcopy(node2)
        #print(new_parent1)
        return new_parent1

    def mutate(self, tree):
        if random.random() < self.mutation_rate:
            nodes = self.get_nodes(tree)
            if nodes:
                node, depth = random.choice(nodes)
                node = self.generate_random_tree(depth)
        return tree

    def tournament_selection(self, losses):
        tournament = random.sample(list(enumerate(losses)), self.tournament_size)
        winner_idx = min(tournament, key=lambda x: x[1])[0]
        return winner_idx

    def fit(self, X, y, verbose=True):
        population = [self.generate_random_tree() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            losses = [self.loss(tree, X, y) for tree in population]
            
            gen_best_loss = min(losses)
            #print(gen_best_loss, max(losses))
            self.loss_history.append(gen_best_loss)
            
            if gen_best_loss < self.best_loss:
                self.best_loss = gen_best_loss
                self.best_solution = deepcopy(population[losses.index(gen_best_loss)])
            
            parents = [population[self.tournament_selection(losses)]
                      for _ in range(self.population_size)]
            
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[min(i + 1, self.population_size - 1)]
                #print(self.get(parent1))
                #print(self.get(parent2))
                temp = deepcopy(parent1)
                child1 = deepcopy(self.crossover(parent1, parent2))
                #print(self.get)
                child2 = deepcopy(self.crossover(parent2, temp))
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                #print(self.get(child1))
                #print(self.get(child2))
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            #print(f"Gen {gen}: {self.best_loss}")
        return self
    
    def get(self, root):
        if root.value in self.operators:
            return "({} {} {})".format(self.get(root.left), root.value, self.get(root.right))
        else:
            return str(root.value)
        
    def visualize_best_solution(self):
        return self.get(self.best_solution)

    def predict(self, X):
        if self.best_solution is None:
            raise ValueError("Model has not been fitted yet.")
        return np.array([self.evaluate_tree(self.best_solution, x) for x in X])
