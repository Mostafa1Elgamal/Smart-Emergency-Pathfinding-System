import pygame
import random
import sys
import time
import heapq

COLOR_BG = (20, 25, 30)
COLOR_PANEL = (40, 45, 50)
COLOR_HIGHWAY = (60, 100, 150)
COLOR_STREET = (100, 100, 100)
COLOR_TRAFFIC = (200, 80, 80)
COLOR_BUILDING = (25, 30, 35)

COLOR_AMBULANCE = (0, 255, 127)
COLOR_PATIENT = (0, 191, 255)
COLOR_DEAD = (80, 80, 80)

COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_ACCENT = (255, 215, 0)

PATH_COLORS = {
    "BFS": (255, 255, 255),       "DFS": (255, 0, 255),
    "UCS": (255, 140, 0),         "IDS": (138, 43, 226),
    "A*": (255, 215, 0),          "Hill": (135, 206, 235),
    "Genetic": (255, 105, 180),   "None": (200, 200, 200)
}
COLOR_VISITED = (100, 255, 100) 

BLOCK_SIZE = 24
GRID_WIDTH = 35
GRID_HEIGHT = 22
MARGIN = 1
DASHBOARD_HEIGHT = 180

COST_HIGHWAY = 1
COST_STREET = 3
COST_TRAFFIC = 15
HEALTH_DECAY_RATE = 8.0

class Node:
    def __init__(self, row, col, type="street"):
        self.row = row
        self.col = col
        self.type = type
        self.cost = COST_STREET
        if type == "highway": self.cost = COST_HIGHWAY
        elif type == "traffic": self.cost = COST_TRAFFIC
        elif type == "building": self.cost = float('inf')
    def get_pos(self): return (self.row, self.col)
    def is_obstacle(self): return self.type == "building"
    def __lt__(self, other): return self.cost < other.cost

class Patient:
    def __init__(self, node):
        self.node = node
        self.health = 100.0
        self.is_dead = False
        self.saved = False
    
    def update_health(self, dt):
        if not self.saved and not self.is_dead:
            decay_amount = HEALTH_DECAY_RATE * dt
            self.health -= decay_amount
            if self.health <= 0: 
                self.health = 0
                self.is_dead = True
                
    def get_pos(self): return self.node.get_pos()

class CityGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = []
        self.start = None
        self.patients = []
        self.generate_city_layout()

    def generate_city_layout(self):
        self.grid = [[Node(r, c, "building") for c in range(self.width)] for r in range(self.height)]
        r = 1
        while r < self.height - 1:
            row_type = "highway" if r % 8 == 1 else "street"
            for c in range(self.width): self.grid[r][c] = Node(r, c, row_type)
            r += random.randint(3, 5)
        c = 1
        while c < self.width - 1:
            col_type = "highway" if c % 10 == 1 else "street"
            for r in range(self.height): 
                if self.grid[r][c].type != "highway": self.grid[r][c] = Node(r, c, col_type)
            c += random.randint(4, 7)
        for r in range(self.height):
            self.grid[r][0] = Node(r, 0, "highway"); self.grid[r][self.width-1] = Node(r, self.width-1, "highway")
        for c in range(self.width):
            self.grid[0][c] = Node(0, c, "highway"); self.grid[self.height-1][c] = Node(self.height-1, c, "highway")
        for _ in range(50):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            if not self.grid[r][c].is_obstacle(): self.grid[r][c] = Node(r, c, "traffic")

    def spawn_random_patients(self, count=1):
        self.patients = []
        roads = [self.grid[r][c] for r in range(self.height) for c in range(self.width) if not self.grid[r][c].is_obstacle()]
        candidates = [n for n in roads if n != self.start]
        if candidates:
            chosen = random.sample(candidates, min(count, len(candidates)))
            for node in chosen: self.patients.append(Patient(node))

    def set_start_point(self):
        roads = [self.grid[r][c] for r in range(self.height) for c in range(self.width) if not self.grid[r][c].is_obstacle()]
        if roads: self.start = random.choice(roads); self.start.type = "street"; self.start.cost = 1

def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def draw_health_bar(screen, x, y, health):
    pygame.draw.rect(screen, (0, 0, 0), (x, y - 6, BLOCK_SIZE, 4))
    if health > 0:
        color = (0, 255, 0) if health > 50 else (255, 165, 0) if health > 20 else (255, 0, 0)
        width = int((health / 100) * BLOCK_SIZE)
        pygame.draw.rect(screen, color, (x, y - 6, width, 4))

def draw_grid(screen, city, font, path=[], visited=set(), visual_time=None, real_time=None, algo_name="None", dt=0.0):
    screen.fill(COLOR_BG)
    
    if dt > 0:
        for p in city.patients: p.update_health(dt)

    for r in range(city.height):
        for c in range(city.width):
            node = city.grid[r][c]
            color = COLOR_BUILDING
            if node.type == "highway": color = COLOR_HIGHWAY
            elif node.type == "street": color = COLOR_STREET
            elif node.type == "traffic": color = COLOR_TRAFFIC
            
            if (r, c) in visited: color = COLOR_VISITED
            if (r, c) in path: color = PATH_COLORS.get(algo_name.split()[0], (255,255,255))
            if city.start and (r, c) == city.start.get_pos(): color = COLOR_AMBULANCE

            x, y = (MARGIN + BLOCK_SIZE) * c + MARGIN, (MARGIN + BLOCK_SIZE) * r + MARGIN
            pygame.draw.rect(screen, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))
            
            if not node.is_obstacle():
                txt = font.render(str(node.cost), True, (255,255,255) if node.cost>5 else (0,0,0))
                screen.blit(txt, (x+8, y+6))

    for p in city.patients:
        pr, pc = p.get_pos()
        px, py = (MARGIN + BLOCK_SIZE) * pc + MARGIN, (MARGIN + BLOCK_SIZE) * pr + MARGIN
        col = COLOR_DEAD if p.is_dead else COLOR_PATIENT
        pygame.draw.circle(screen, col, (px + BLOCK_SIZE//2, py + BLOCK_SIZE//2), BLOCK_SIZE//3)
        draw_health_bar(screen, px, py, p.health)

    ui_y = (BLOCK_SIZE + MARGIN) * city.height + 5
    pygame.draw.rect(screen, COLOR_PANEL, (0, ui_y, screen.get_width(), DASHBOARD_HEIGHT))
    
    status = f"ALGO: {algo_name}"
    if visual_time: status += f" | Vis: {visual_time}s"
    if real_time: status += f" | Real: {real_time}s"
    screen.blit(font.render(status, True, COLOR_ACCENT), (20, ui_y + 15))

    alive = sum(1 for p in city.patients if not p.is_dead)
    p_info = f"Patients Alive: {alive}/{len(city.patients)}"
    screen.blit(font.render(p_info, True, (0, 255, 0) if alive==len(city.patients) else (255, 0, 0)), (20, ui_y + 40))

    instr = [
        "CONTROLS:",
        "[Shift]+Key: Instant Result | [M]: Random 5 Patients",
        "[B] BFS | [D] DFS | [U] UCS | [I] IDS",
        "[A] A* | [H] Hill | [G] Genetic (TSP)",
        "[R] Reset Map | [P] New Start/Goal | [C] Clear"
    ]
    for i, t in enumerate(instr):
        screen.blit(font.render(t, True, (200, 200, 200)), (450, ui_y + 15 + i*20))

def visual_bfs(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    
    t0 = time.time()
    queue = [[start]]
    visited = {start}
    
    clock = pygame.time.Clock()
    
    while queue:
        dt = clock.tick(60) / 1000.0 
        
        path = queue.pop(0)
        node = path[-1]
        
        if node == goal:
            return path, round(time.time()-t0, 3)
        
        r, c = node
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in visited:
                visited.add((nr,nc))
                new_path = list(path)
                new_path.append((nr,nc))
                queue.append(new_path)
        
        draw_grid(screen, city, font, [], visited, algo_name="BFS", dt=dt)
        pygame.display.flip()
    return [], 0

def real_bfs(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); q = [[start]]; v = {start}
    while q:
        path = q.pop(0); node = path[-1]
        if node == goal: return path, round(time.time()-t0, 4)
        r, c = node
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in v:
                v.add((nr,nc)); q.append(path + [(nr,nc)])
    return [], 0

def visual_dfs(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    
    t0 = time.time()
    stack = [[start]]
    visited = {start}
    clock = pygame.time.Clock()
    
    while stack:
        dt = clock.tick(60) / 1000.0
        path = stack.pop()
        node = path[-1]
        
        if node == goal:
            return path, round(time.time()-t0, 3)
        
        r, c = node
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in visited:
                visited.add((nr,nc))
                new_path = list(path)
                new_path.append((nr,nc))
                stack.append(new_path)
        
        draw_grid(screen, city, font, [], visited, algo_name="DFS", dt=dt)
        pygame.display.flip()
    return [], 0

def real_dfs(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); stack = [[start]]; v = {start}
    while stack:
        path = stack.pop(); node = path[-1]
        if node == goal: return path, round(time.time()-t0, 4)
        r, c = node
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in v:
                v.add((nr,nc)); stack.append(path + [(nr,nc)])
    return [], 0

def visual_ucs(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    
    t0 = time.time()
    pq = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()
    clock = pygame.time.Clock()
    
    while pq:
        dt = clock.tick(60) / 1000.0
        _, current = heapq.heappop(pq)
        visited.add(current)
        
        if current == goal:
            path = []; curr = current
            while curr in came_from: path.append(curr); curr = came_from[curr]
            path.append(start); path.reverse()
            return path, round(time.time()-t0, 3)
        
        r, c = current
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle():
                new_cost = cost_so_far[current] + city.grid[nr][nc].cost
                if (nr,nc) not in cost_so_far or new_cost < cost_so_far[(nr,nc)]:
                    cost_so_far[(nr,nc)] = new_cost
                    heapq.heappush(pq, (new_cost, (nr,nc)))
                    came_from[(nr,nc)] = current
        
        draw_grid(screen, city, font, [], visited, algo_name="UCS", dt=dt)
        pygame.display.flip()
    return [], 0

def real_ucs(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); pq = [(0, start)]; came = {}; costs = {start: 0}
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == goal:
            path = []; c = curr
            while c in came: path.append(c); c = came[c]
            path.append(start); path.reverse()
            return path, round(time.time()-t0, 4)
        for nr, nc in [(curr[0]-1,curr[1]), (curr[0]+1,curr[1]), (curr[0],curr[1]-1), (curr[0],curr[1]+1)]:
             if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle():
                ncost = costs[curr] + city.grid[nr][nc].cost
                if (nr,nc) not in costs or ncost < costs[(nr,nc)]:
                    costs[(nr,nc)] = ncost; heapq.heappush(pq, (ncost, (nr,nc))); came[(nr,nc)] = curr
    return [], 0

def dls(city, node, goal, limit, path, visited):
    path.append(node); visited.add(node)
    if node == goal: return True
    if limit <= 0: path.pop(); return False
    r, c = node
    for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
        if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in visited:
            if dls(city, (nr,nc), goal, limit-1, path, visited): return True
    path.pop(); return False

def visual_ids(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); depth = 0
    clock = pygame.time.Clock()
    
    while True:
        visited = set(); path = []
        dt = clock.tick(60) / 1000.0
        
        if dls(city, start, goal, depth, path, visited):
            return path, round(time.time()-t0, 3)
        
        draw_grid(screen, city, font, [], visited, algo_name="IDS", dt=dt)
        pygame.display.flip()
        depth += 1
        if depth > 100: return [], 0

def real_ids(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); depth = 0
    while True:
        visited = set(); path = []
        if dls(city, start, goal, depth, path, visited):
            return path, round(time.time()-t0, 4)
        depth += 1; 
        if depth > 100: return [], 0

def visual_astar(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    
    t0 = time.time()
    pq = [(0, start)]
    came_from = {}
    g_cost = {start: 0}
    visited = set()
    clock = pygame.time.Clock()
    
    while pq:
        dt = clock.tick(60) / 1000.0
        _, current = heapq.heappop(pq)
        visited.add(current)
        
        if current == goal:
            path = []; curr = current
            while curr in came_from: path.append(curr); curr = came_from[curr]
            path.append(start); path.reverse()
            return path, round(time.time()-t0, 3)
        
        r, c = current
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle():
                new_cost = g_cost[current] + city.grid[nr][nc].cost
                if (nr,nc) not in g_cost or new_cost < g_cost[(nr,nc)]:
                    g_cost[(nr,nc)] = new_cost
                    priority = new_cost + heuristic((nr,nc), goal)
                    heapq.heappush(pq, (priority, (nr,nc)))
                    came_from[(nr,nc)] = current
        
        draw_grid(screen, city, font, [], visited, algo_name="A*", dt=dt)
        pygame.display.flip()
    return [], 0

def real_astar(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); pq = [(0, start)]; came = {}; g = {start: 0}
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == goal:
            path = []; c = curr
            while c in came: path.append(c); c = came[c]
            path.append(start); path.reverse()
            return path, round(time.time()-t0, 4)
        for nr, nc in [(curr[0]-1,curr[1]), (curr[0]+1,curr[1]), (curr[0],curr[1]-1), (curr[0],curr[1]+1)]:
            if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle():
                ncost = g[curr] + city.grid[nr][nc].cost
                if (nr,nc) not in g or ncost < g[(nr,nc)]:
                    g[(nr,nc)] = ncost; heapq.heappush(pq, (ncost+heuristic((nr,nc), goal), (nr,nc))); came[(nr,nc)] = curr
    return [], 0

def visual_hill(city, screen, font):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    
    t0 = time.time()
    current = start
    path = [current]
    visited = {current}
    clock = pygame.time.Clock()
    
    while current != goal:
        dt = clock.tick(15) / 1000.0
        
        r, c = current
        neighbors = [(nr,nc) for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] 
                     if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in visited]
        if not neighbors: break
        
        next_node = min(neighbors, key=lambda x: heuristic(x, goal))
        current = next_node
        path.append(current)
        visited.add(current)
        
        draw_grid(screen, city, font, [], visited, algo_name="Hill", dt=dt)
        pygame.display.flip()
        
    return path, round(time.time()-t0, 3)

def real_hill(city):
    if not city.patients: return [], 0
    start, goal = city.start.get_pos(), city.patients[0].get_pos()
    t0 = time.time(); curr = start; path = [curr]; v = {curr}
    while curr != goal:
        neighbors = [(nr,nc) for nr,nc in [(curr[0]-1,curr[1]),(curr[0]+1,curr[1]),(curr[0],curr[1]-1),(curr[0],curr[1]+1)] 
                     if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle() and (nr,nc) not in v]
        if not neighbors: break
        curr = min(neighbors, key=lambda x: heuristic(x, goal))
        path.append(curr); v.add(curr)
    return path, round(time.time()-t0, 4)

def get_segment_path(city, start, end):
    pq=[(0,start)]; came={}; g={start:0}
    while pq:
        _,u=heapq.heappop(pq)
        if u==end:
            p=[]; c=u
            while c in came: p.append(c); c=came[c]
            p.append(start); p.reverse(); return p
        for nr,nc in [(u[0]-1,u[1]),(u[0]+1,u[1]),(u[0],u[1]-1),(u[0],u[1]+1)]:
             if 0<=nr<city.height and 0<=nc<city.width and not city.grid[nr][nc].is_obstacle():
                ncost=g[u]+city.grid[nr][nc].cost
                if (nr,nc) not in g or ncost<g[(nr,nc)]:
                    g[(nr,nc)]=ncost; heapq.heappush(pq,(ncost+heuristic((nr,nc),end),(nr,nc))); came[(nr,nc)]=u
    return []

def calc_tsp_dist(order, city):
    dist=0; curr=city.start.get_pos()
    for i in order:
        target=city.patients[i].get_pos()
        dist+=abs(curr[0]-target[0])+abs(curr[1]-target[1])
        curr=target
    return dist

def construct_tsp_path(order, city):
    full=[]; curr=city.start.get_pos()
    for i in order:
        target=city.patients[i].get_pos()
        seg=get_segment_path(city, curr, target)
        if seg: full.extend(seg[:-1])
        curr=target
    full.append(curr)
    return full

def visual_genetic(city, screen, font):
    t0 = time.time()
    n = len(city.patients)
    if n < 2: return [], 0
    
    indices = list(range(n))
    pop = [random.sample(indices, n) for _ in range(20)]
    best_path = []
    clock = pygame.time.Clock()
    
    for gen in range(50):
        dt = clock.tick(10) / 1000.0 
        
        pop.sort(key=lambda p: calc_tsp_dist(p, city))
        best_path = construct_tsp_path(pop[0], city)
        
        draw_grid(screen, city, font, path=best_path, algo_name="Genetic", dt=dt)
        pygame.display.flip()
        
        survivors = pop[:10]; new_pop = list(survivors)
        while len(new_pop) < 20:
            child = list(random.choice(survivors))
            if random.random() < 0.4:
                a, b = random.sample(range(n), 2)
                child[a], child[b] = child[b], child[a]
            new_pop.append(child)
        pop = new_pop
        
    for p in city.patients: p.saved = True
    return best_path, round(time.time()-t0, 3)

def real_genetic(city):
    t0 = time.time(); n = len(city.patients)
    if n < 2: return [], 0
    pop = [random.sample(range(n), n) for _ in range(20)]
    for _ in range(50):
        pop.sort(key=lambda p: calc_tsp_dist(p, city))
        survivors = pop[:10]; new_pop = list(survivors)
        while len(new_pop) < 20:
            c = list(random.choice(survivors))
            if random.random() < 0.3:
                a, b = random.sample(range(n), 2); c[a],c[b]=c[b],c[a]
            new_pop.append(c)
    return construct_tsp_path(pop[0], city), round(time.time()-t0, 4)

def main():
    pygame.init()
    font = pygame.font.SysFont('consolas', 14, bold=True)
    W = ((BLOCK_SIZE + MARGIN) * GRID_WIDTH) + MARGIN
    H = ((BLOCK_SIZE + MARGIN) * GRID_HEIGHT) + MARGIN + DASHBOARD_HEIGHT
    screen = pygame.display.set_mode([W, H])
    pygame.display.set_caption("S.E.R.S - Smart Emergency Response System")
    clock = pygame.time.Clock()
    
    city = CityGrid(GRID_WIDTH, GRID_HEIGHT)
    city.set_start_point()
    city.spawn_random_patients(1)
    
    path = []; v_time = None; r_time = None; algo = "System Idle"
    running = True
    
    while running:
        clock.tick(60) 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.KEYDOWN:
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                path = []; v_time = None
                
                for p in city.patients: 
                    if not p.is_dead: p.health = 100.0; p.saved = False

                if event.key == pygame.K_b:
                    algo = "BFS"
                    if shift: path, r_time = real_bfs(city)
                    else: path, v_time = visual_bfs(city, screen, font); _, r_time = real_bfs(city)

                elif event.key == pygame.K_d:
                    algo = "DFS"
                    if shift: path, r_time = real_dfs(city)
                    else: path, v_time = visual_dfs(city, screen, font); _, r_time = real_dfs(city)

                elif event.key == pygame.K_u:
                    algo = "UCS"
                    if shift: path, r_time = real_ucs(city)
                    else: path, v_time = visual_ucs(city, screen, font); _, r_time = real_ucs(city)

                elif event.key == pygame.K_i:
                    algo = "IDS"
                    if shift: path, r_time = real_ids(city)
                    else: path, v_time = visual_ids(city, screen, font); _, r_time = real_ids(city)

                elif event.key == pygame.K_a:
                    algo = "A*"
                    if shift: path, r_time = real_astar(city)
                    else: path, v_time = visual_astar(city, screen, font); _, r_time = real_astar(city)

                elif event.key == pygame.K_h:
                    algo = "Hill"
                    if shift: path, r_time = real_hill(city)
                    else: path, v_time = visual_hill(city, screen, font); _, r_time = real_hill(city)

                elif event.key == pygame.K_g:
                    algo = "Genetic"
                    if len(city.patients) < 5: city.spawn_random_patients(5)
                    if shift: path, r_time = real_genetic(city)
                    else: path, v_time = visual_genetic(city, screen, font); _, r_time = real_genetic(city)

                elif event.key == pygame.K_m:
                    city.spawn_random_patients(5); algo = "New Signal: 5 Patients"; path=[]
                elif event.key == pygame.K_r:
                    city = CityGrid(GRID_WIDTH, GRID_HEIGHT); city.set_start_point(); city.spawn_random_patients(1); algo = "Reboot"
                elif event.key == pygame.K_p:
                    city.set_start_point(); city.spawn_random_patients(1); algo = "New Coords"
                elif event.key == pygame.K_c:
                    path = []; algo = "Cleared"

        draw_grid(screen, city, font, path, set(), v_time, r_time, algo, dt=0)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()