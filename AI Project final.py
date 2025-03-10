# For game Section
import pygame
# For genetic algorithm 
from deap import base, creator, tools, algorithms
import numpy as np

# Game level dimensions
LEVEL_WIDTH = 20 
LEVEL_HEIGHT = 15

# Define the genome and fitness function
def evaluate(individual):
    level = np.array(individual).reshape((LEVEL_HEIGHT, LEVEL_WIDTH))
    
    start = (0, 0)
    goal = (LEVEL_HEIGHT - 1, LEVEL_WIDTH - 1)

    # BFS fitness
    if not is_reachable(level, start, goal):
        return 0,  # Strong penalty for unplayable levels
    
    empty_cells = np.sum(level == 0)
    print("Empty Cell: ", empty_cells)

    # fitness
    fitness = (empty_cells / (LEVEL_WIDTH * LEVEL_HEIGHT)) * 100

    return fitness,

def get_neighbors(pos):
    neighbors = [(pos[0] + dx, pos[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    return [(x, y) for x, y in neighbors if 0 <= x < LEVEL_HEIGHT and 0 <= y < LEVEL_WIDTH]

# BFS to check if the goal is reachable from start.
def is_reachable(level, start, goal):
    queue = [start]
    visited = set()
    visited.add(start)
    
    while queue:
        current = queue.pop(0)
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            if level[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False

probability_0 = 0.6

def generate_individual():
    return np.random.choice([0, 1], size=1, p=[probability_0, 1 - probability_0])

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", generate_individual)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=LEVEL_WIDTH * LEVEL_HEIGHT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# CrossOver => TwoPoint
toolbox.register("mate", tools.cxTwoPoint)
# Mutation => Inversion
toolbox.register("mutate", tools.mutInversion)
# Selection => Tournament
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)

# Genetic Algorithm parameters
population = toolbox.population(n=100)
cxpb = 0.5
mutpb = 0.5
FitnessTres = 75
MAX_GENERATIONS = 150

# fitness function = (Number of Empty Cells / (Level Width * Level height)) * 100

# Run the genetic algorithm with a retry mechanism for ensuring completeness
def run_evolution():
    for gen in range(MAX_GENERATIONS):
        print(f"Generation: {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))
        
        # Check for a valid level in the population
        best_ind = tools.selBest(population, 1)[0]
        if best_ind.fitness.values[0] >= FitnessTres:
            return best_ind
    return None

best_ind = run_evolution()
if best_ind is None:
    #best_ind = run_evolution()
    raise Exception("Failed to find a valid level within the maximum number of generations")

best_level = np.array(best_ind).reshape((LEVEL_HEIGHT, LEVEL_WIDTH))
print(f'Best Individual: \n{best_level}')

move_delay = 200  # Milliseconds between moves
last_move_time = pygame.time.get_ticks()
victory_time = None
path = []

# BFS to find the path from start to goal
def find_path(start, goal, level):
    queue = [(start, [start])]
    visited = set()
    visited.add(start)
    
    while queue:
        (current, path) = queue.pop(0)
        for neighbor in get_neighbors(current):
            if neighbor == goal:
                return path + [neighbor]
            if level[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return []


# Initialize pygame
pygame.init()

CELL_SIZE = 30

# Set Window
screen = pygame.display.set_mode((600, 450)) 
clock = pygame.time.Clock()
font = pygame.font.Font(None, 74)

# Load player sprite and fit to the CELL_SIZE / 2
player_pos = [0, 0]
player_radius = CELL_SIZE // 2

# Load obstacle sprite and fit to the CELL_SIZE
obstacle_image = pygame.image.load('obstacle.png')
obstacle_image = pygame.transform.scale(obstacle_image, (CELL_SIZE, CELL_SIZE))

COLOR_START = (0, 255, 0)      # green color
COLOR_END = (255, 0, 0)        # red coor
COLOR_EMPTY = (255, 255, 255)  # white color

# Pygame loop to render the generated content and handle player movement
running = True
automatic_mode = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_time = pygame.time.get_ticks()
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_p] and not automatic_mode:
        path = find_path(tuple(player_pos), (LEVEL_HEIGHT - 1, LEVEL_WIDTH - 1), best_level)
        automatic_mode = True

    if automatic_mode and path:
        if current_time - last_move_time > move_delay:
            next_pos = path.pop(0)
            player_pos = [next_pos[0], next_pos[1]]
            last_move_time = pygame.time.get_ticks()

            if player_pos == [LEVEL_HEIGHT - 1, LEVEL_WIDTH - 1]:
                victory_text = font.render("Victory!", True, (0, 0, 0))
                screen.blit(victory_text, (200, 150))
                pygame.display.flip()
                pygame.time.wait(2000)  # Display for 2 seconds
                running = False

    # Render the best individual
    for y in range(LEVEL_HEIGHT):

        for x in range(LEVEL_WIDTH):

            if (y, x) == (0, 0):
                color = COLOR_START

            elif (y, x) == (LEVEL_HEIGHT - 1, LEVEL_WIDTH - 1):
                color = COLOR_END

            elif best_level[y, x] == 1:
                screen.blit(obstacle_image, (x * CELL_SIZE, y * CELL_SIZE))
                continue

            else:
                color = COLOR_EMPTY

            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw the player
    pygame.draw.circle(screen, (0, 0, 255), 
                      (player_pos[1] * CELL_SIZE + player_radius, player_pos[0] * CELL_SIZE + player_radius), 
                       player_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()