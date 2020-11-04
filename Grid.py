
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return 'x: ' + str(self.x) + ' y: ' + str(self.y)

    def __hash__(self):
        return hash((self.x, self.y))



class Grid:
    def __init__(self, numRows, numCols, walls, terminal_states, reward, transition, discount_rate, epsilon):
        self.numRows = numRows
        self.numCols = numCols
        self.walls = walls
        self.terminal_states = terminal_states
        self.reward = reward
        self.transition = transition
        self.discount_rate = discount_rate
        self.epsilon = epsilon


    def moves(self, state, action):
        if action == 'E':
            if state.x < self.numCols:
                if State(state.x+1, state.y) not in self.walls:
                    return State(state.x+1, state.y)

        if action == 'W':
            if state.x > 1:
                if State(state.x-1, state.y) not in self.walls:
                    return State(state.x-1, state.y)

        if action == 'N':
            if state.y < self.numRows:
                if State(state.x, state.y+1) not in self.walls:
                    return State(state.x, state.y+1)

        if action == 'S':
            if state.y > 1:
                if State(state.x, state.y-1) not in self.walls:
                    return State(state.x, state.y-1)

        return state



def readFile(inputFile):
    grid = {}
    file = open(inputFile, 'r')
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':')
            grid[key.strip()] = value.strip()
    return grid
