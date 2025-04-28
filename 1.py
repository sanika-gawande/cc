#dfs
'''def dfs(graph,start):
    visited=[]
    stack=[start]
    
    print("DFS Traversal : ")
    while stack:
        node=stack.pop(0)
        if node not in visited:
            print(node,end=" ")
            visited.append(node)
            stack.extend(reversed(graph.get(node,[])))
            
graph={}
print("Build graph: ")
while True:
    node=input("Enter node(leave empty to stop ")
    if not node:
        break
    neighbour=input(f"Enter neigbour{node} ").split(',')
    graph[node]=[n.strip() for n in neighbour]
    
start_node=input("Enter start node ")
dfs(graph,start_node)


# bfs
# Simple BFS (Breadth-First Search)
def bfs(graph, start):
    visited = []
    queue = [start]

    print("\nBFS Traversal:", end=" ")
    while queue:
        node = queue.pop(0)
        if node not in visited:
            print(node, end=" ")
            visited.append(node)
            queue.extend(graph.get(node, []))

# Build Graph
graph = {}
print("Create the graph (for BFS):")
while True:
    node = input("Enter node (leave empty to stop): ")
    if not node:
        break
    neighbors = input(f"Enter neighbors of {node} (comma separated): ").split(',')
    graph[node] = [n.strip() for n in neighbors]

start_node = input("\nEnter starting node for BFS: ")
bfs(graph, start_node)'''


// prims algo...

import sys

def min_pst(graph, v):
    selected = [False] * v
    selected[0] = True
    print("\nEdges \tWeight")

    for _ in range(v - 1):
        min_w = sys.maxsize
        x = y = 0

        for i in range(v):
            if selected[i]:
                for j in range(v):
                    if not selected[j] and graph[i][j]:
                        if graph[i][j] < min_w:
                            min_w = graph[i][j]
                            x, y = i, j

        print(f"{x} - {y}\t{min_w}")
        selected[y] = True

# --- User Input Part ---

V = int(input("Enter number of vertices: "))
E = int(input("Enter number of edges: "))

# Initialize graph matrix with 0s
graph = [[0 for _ in range(V)] for _ in range(V)]

print("\nEnter edges in format: u v weight")
for _ in range(E):
    u, v2, w = map(int, input().split())
    graph[u][v2] = w
    graph[v2][u] = w  # because the graph is undirected

min_pst(graph, V)

# Simple Kruskal's Algorithm

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

# Main
n = int(input("Enter number of vertices: "))
vertices = []
print("Enter vertex names:")
for _ in range(n):
    vertices.append(input().strip())

e = int(input("Enter number of edges: "))
edges = []
print("Enter edges (source destination weight):")
for _ in range(e):
    u, v, w = input().split()
    edges.append((int(w), u, v))  # store weight first for easy sorting

# Kruskal's algorithm
edges.sort()
parent = {v: v for v in vertices}
rank = {v: 0 for v in vertices}

mst = []
cost = 0

for w, u, v in edges:
    if find(parent, u) != find(parent, v):
        union(parent, rank, u, v)
        mst.append((u, v, w))
        cost += w

print("\nEdges in MST:")
for u, v, w in mst:
    print(f"{u} - {v} : {w}")

print(f"Total cost: {cost}")




# N-Queens Problem using Backtracking and Branch and Bound

def is_safe(board, row, col, n):
    # Check column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # Check upper-right diagonal
    i, j = row, col
    while i >= 0 and j < n:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True

def solve_n_queens(board, row, n):
    if row == n:
        # All queens placed successfully
        return True

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            if solve_n_queens(board, row + 1, n):
                return True
            # Backtrack
            board[row][col] = 0

    return False

def print_board(board, n):
    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()

# Main
n = int(input("Enter the value of N for N-Queens: "))

board = [[0 for _ in range(n)] for _ in range(n)]

if solve_n_queens(board, 0, n):
    print("\nOne solution for", n, "Queens:")
    print_board(board, n)
else:
    print("No solution exists for", n, "Queens.")



#chatbot
import time

def greet_user():
    print("Hello! Welcome to our Customer Service Chatbot.")
    time.sleep(1)
    print("\n How can I assist You today ?")
    time.sleep(1)
    
def choice():
    print("please Select an opion: ")
    print("\n 1. Product Information ")
    print("\n 2. Track my order ")
    print("\n 3. File a complaint ")
    print("\n 4. Exist ")
    

def product_Info():
    print("We have variety of products:")
    time.sleep(1)
    print("-smartphones \n -laptops \n For more Details visit our website.")
    time.sleep(2)
    
def track_order():
    print("Please enter your order ID to track your order: ")
    order=input("order_id: ")
    
    print(f" Tracking your order with ID {order}")
    time.sleep(2)
    
    print(f"Order {order} is on its way! It will be delivered in 2 days ")
    
def file_com():
    print("We are sorry to hear you have an issue. ")
    time.sleep(1)
    input("Please describe your complaint: ")
    print("Thank you for your feedback. Our support team will reach out to you soon. ")
    
def exit():
    print("Thank you for chatting with us. Have a great day! ")
    

def chatbot():
    greet_user()
    
    while True:
        choice()
        user_choice=input("Enter your choice")
        if user_choice=="1":
            product_Info()
        elif user_choice=="2":
            track_order()
        elif user_choice == "3":
            file_com()
        elif user_choice=="4":
            exit()
            break
        else :
            print("sorry in valid option :")
        
        
if __name__=="__main__":
    chatbot()



// A* 

import heapq

# A* Algorithm for Game Search
def a_star(grid, start, goal):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    open_list = [(0, start)]  # Priority Queue: (f_score, node)
    g_score = {start: 0}  # Cost from start to node
    came_from = {}  # For reconstructing path

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return [start] + path[::-1]

        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if move is valid (inside grid and not obstacle)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                new_cost = g_score[current] + 1
                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_cost
                    f_score = new_cost + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])  # Manhattan distance
                    heapq.heappush(open_list, (f_score, neighbor))

    return None  # No path found

# Main code
if _name_ == "_main_":
    # Input grid size
    rows, cols = map(int, input("Enter grid size (rows cols): ").split())

    # Input grid
    print(f"Enter the grid ({rows} rows with {cols} columns each, 0 for free, 1 for obstacle):")
    grid = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            print("Invalid row size! Please re-enter.")
            exit()
        grid.append(row)

    # Input start and goal positions
    start = tuple(map(int, input("Enter start position (row col): ").split()))
    goal = tuple(map(int, input("Enter goal position (row col): ").split()))

    # Check if start and goal are valid
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        print("Invalid start or goal position!")
        exit()
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("Start or goal is an obstacle!")
        exit()

    # Run A* and display result
    path = a_star(grid, start, goal)
    if path:
        print("\nPath found:")
        for step in path:
            print(step)
    else:
        print("\nNo path found!")
            
            
