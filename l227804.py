#q1

from collections import deque

def find_shortest_path(matrix):
    # Directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Initialize starting and ending positions
    start = (0, 0)  # Assuming start is (1, 1) based on your description
    end = (len(matrix) - 1, len(matrix[0]) - 1)  # Assuming end is (4, 4) and matrix is square

    # Initialize data structures for BFS
    queue = deque([(start, [start])]) #deque for BFS
    visited = set([start])  # Track visited positions

    # BFS Loop
    while queue:
        (row, col), path = queue.popleft()  # Dequeue the next position and path

        # Check if HOME is reached
        if (row, col) == end:
            return path  

        # Mark the position as visited
        visited.add((row, col))

        # Explore neighboring cells (non-diagonal and not obstacles)
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check 
            if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]) and \
               matrix[new_row][new_col] != 0 and (new_row, new_col) not in visited:
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))

    return None

#q2

import time

def state_to_tuple(state):
    """Convert a string state to a tuple representation."""
    state_list = list(state)
    matrix = [state_list[i:i+3] for i in range(0, len(state_list), 3)]
    return tuple(map(tuple, matrix))

def tuple_to_state(matrix):
    """Convert a tuple representation back to a string state."""
    state = ''.join([''.join(row) for row in matrix])
    return state

def get_moves(matrix, moves_dict={}):
    """Generate possible moves from the given state, using a dictionary for optimization."""
    state_tuple = tuple(map(tuple, matrix)) 
    if state_tuple in moves_dict:
        return moves_dict[state_tuple]  

    moves = []
    row, col = next((r, c) for r, row in enumerate(matrix) for c, val in enumerate(row) if val == '0')
    
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_matrix = [list(row) for row in matrix]
            new_matrix[row][col], new_matrix[new_row][new_col] = new_matrix[new_row][new_col], new_matrix[row][col]
            moves.append(tuple(map(tuple, new_matrix)))
            
    moves_dict[state_tuple] = moves  
    return moves

def dfs(start_state, goal_state):
    """Perform Depth-First Search (DFS) with optimizations."""
    stack = [(start_state, [start_state])]
    visited = {start_state}
    solution_found = False 

    while stack and not solution_found:  
        current_state, path = stack.pop()

        if current_state == goal_state:
            return path, True  

        for next_state in get_moves(current_state):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [next_state]))

    return None, False  

def main():
    """Main function to take input and execute the optimized DFS algorithm."""
    start_state = input("Enter start State: ")
    goal_state = input("Enter goal State: ")
    start_tuple = state_to_tuple(start_state)
    goal_tuple = state_to_tuple(goal_state)
    print("-----------------")
    print("DFS Algorithm")
    print("-----------------")
    
    start_time = time.time()
    solution_path, solution_found = dfs(start_tuple, goal_tuple)
    end_time = time.time()
    
    if solution_found:
        print("Time taken:", end_time - start_time, "seconds")
        print("Path Cost:", len(solution_path) -1)
        print("No of Node Visited:", len(solution_path))
        for state in solution_path:
            for row in state:
                print(' '.join(row))
            print("-----")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()

#q3

from collections import deque

class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {
            'The': 4,
            'cat': 3,
            'dog': 3,
            'runs': 2,
            'fast': 1 # Goal node heuristic should be 0
        }
        return H.get(n, float('inf'))
    
    def reconst_path(self, parents, current):
        total_path = [current]
        while current in parents and parents[current] != current:
            current = parents[current]
            total_path.append(current)
        return total_path[::-1]
    
    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set()
        g = {start_node: 0} # Cost from start node to all other nodes
        parents = {start_node: start_node} # Keeps track of paths

        while open_list:
            n = min(open_list, key=lambda node: g[node] + self.h(node))
            
            if n == stop_node:
                path = self.reconst_path(parents, stop_node)
                total_cost = sum(
                    next(weight for neighbor, weight in self.adjacency_list[parent] if neighbor == child)
                    for parent, child in zip(path[:-1], path[1:])
                )
                print(f"Sentence: {' '.join(path)}")
                print(f"Total cost: {total_cost}")
                return path
            
            open_list.remove(n)
            closed_list.add(n)

            for (m, weight) in self.get_neighbors(n):
                if m in closed_list:
                    continue
                tentative_g = g[n] + weight
                
                if m not in open_list or tentative_g < g.get(m, float('inf')):
                    g[m] = tentative_g
                    parents[m] = n
                    open_list.add(m)
        
        print("Path does not exist!")
        return None

adjacency_list = {
    'The': [('cat', 2), ('dog', 3)],
    'cat': [('runs', 1)],
    'dog': [('runs', 2)],
    'runs': [('fast', 2)],
    'fast': []
}

graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('The', 'fast')
