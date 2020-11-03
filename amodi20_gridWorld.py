import traceback


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



# Parsing the input file.
def parseInput(inputFile):
    # Dictionary for storing the grid world.
    print('-----------------------------PARSING THE INPUT FILE-----------------------------')
    gridWorld = {}
    wallList = []
    terminalStateDict = {}
    file = open(inputFile, 'r')
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':')
            gridWorld[key.strip()] = value.strip()
    # print(gridWorld)

    numRows = gridWorld['size'].split(' ')[1]
    numCols = gridWorld['size'].split(' ')[0]
    walls = gridWorld['walls'].split(',')
    terminalStates = gridWorld['terminal_states'].split(',')
    rewards = gridWorld['reward']
    transition = gridWorld['transition_probabilities']
    discountFactor = gridWorld['discount_rate']
    epsilon = gridWorld['epsilon']

    for wall in walls:
        wallList.append(State(int(wall.strip().split(' ')[0]), int(wall.strip().split(' ')[1])))
    for terminalState in terminalStates:
        x, y, reward = terminalState.strip().split(' ')
        terminalStateDict[State(int(x), int(y))] = float(reward)

    print('Number of Rows: ' + str(int(numRows)) + '\n',
          'Number of Columns: ' + str(int(numCols)) + '\n',
          'Walls: ' + str(wallList) + '\n',
          'Terminal States: ' + str(terminalStateDict) + '\n',
          'Rewards: ' + str(float(rewards)) + '\n',
          'Transition: ' + str(transition) + '\n',
          'Float: ' + str(float(discountFactor)) + '\n',
          'Epsilon: ' + str(float(epsilon)))
    return Grid(int(numRows), int(numCols), wallList, terminalStateDict, float(rewards), transition, float(discountFactor), float(epsilon))



def generateMDP(gridWorld):
    print('-----------------------------GENERATING MDP-----------------------------')
    actionList = ['N', 'E', 'S', 'W']
    states = []
    rewardsDict = {}
    transitionDict = {}
    for i in range(gridWorld.numRows):
        for j in range(gridWorld.numCols):
            states.append(State(j+1, i+1))

    for state in states:
        if state in gridWorld.terminal_states:
            rewardsDict[state] = gridWorld.terminal_states[state]
        else:
            rewardsDict[state] = gridWorld.reward

    for state in states:
        transitionDict[state] = {}
        for action in actionList:
            leftAction = actionList[(actionList.index(action)-1) % len(actionList)]
            rightAction = actionList[(actionList.index(action)+1) % len(actionList)]
            transitionDict[state][action] = [(0.8, gridWorld.moves(state, action)), (0.1, gridWorld.moves(state, leftAction)), (0.1, gridWorld.moves(state, rightAction))]

            if state in gridWorld.terminal_states:
                transitionDict[state][action] = [(0.0, gridWorld.moves(state, action)), (0.0, gridWorld.moves(state, leftAction)), (0.0, gridWorld.moves(state, rightAction))]

    return MDP(states, actionList, transitionDict, rewardsDict, gridWorld.discount_rate, gridWorld.epsilon, gridWorld)



def printGrid(value, numRows, numCols):
    for i in range(numRows, 0, -1):
        for j in range(1, numCols+1):
            if State(j, i) not in value:
                print('- ')
            else:
                print(str(value[State(j, i)]) + ' ')
        print('\n')
    pass




class MDP:
    def __init__(self, states, actions, transitions, rewards, discount_rate, epsilon, gridWorld):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.gridWorld = gridWorld

    def valueIteration(self):
        print('---------------------------VALUE ITERATION---------------------------')
        iteration = 0
        utility = {}
        policy = {}
        for state in self.states:
            if state in self.gridWorld.walls:
                utility[state] = '-----------'
            else:
                utility[state] = 0

        while True:
            print('Iteration Number: ' + str(iteration))
            printGrid(utility, self.gridWorld.numRows, self.gridWorld.numCols)
            utilityCopy = utility.copy()
            delta = 0
            for state in self.states:
                if state in self.gridWorld.walls:
                    continue
                immReward = self.rewards[state]
                maxReward = -99999
                policy[state] = 'T'
                if state in self.gridWorld.terminal_states:
                    maxReward = 0
                else:
                    for action in self.actions:
                        rewardForCurrentAction = 0
                        for transition in self.transitions[state][action]:
                            prob = transition[0]
                            utilityNextState = utilityCopy[transition[1]]
                            rewardForCurrentAction += prob * utilityNextState
                        if rewardForCurrentAction > maxReward:
                            maxReward = rewardForCurrentAction
                            policy[state] = action

                utility[state] = immReward + self.discount_rate * maxReward
                if abs(utility[state] - utilityCopy[state]) > delta:
                    delta = abs(utility[state] - utilityCopy[state])
            if delta <= self.epsilon * (1-self.discount_rate) / self.discount_rate:
                print('Final Value')
                printGrid(utility, self.gridWorld.numRows, self.gridWorld.numCols)
                print('Final Policy')
                print(policy, self.gridWorld.numRows, self.gridWorld.numCols)
                break
            iteration = iteration + 1



if __name__ == '__main__':
    inputFile = 'mdp_input.txt'
    gridWorld = parseInput(inputFile)
    gridMDP = generateMDP(gridWorld)
    gridMDP.valueIteration()


