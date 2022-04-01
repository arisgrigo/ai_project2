# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        ghostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currPos = currentGameState.getPacmanPosition()
        x, y = newPos
        foodList = newFood.asList()
        foodDist = []       # keeps track of each food's manhattan distance from pacman
        closestFood = 0
        foodBonus = 0
        ghostDist = []      # keeps track of each ghost's manhattan distance from pacman
        endScore = 0

        # get manhattan distance from pacman to each ghost
        # if new position is same as that of a ghost return -10000 (because pacman will surely lose)
        for ghost in ghostPos:
            x1, y1 = ghost
            ghostDist.append(abs(x1-x) + abs(y1-y))
            if(x1 == x and y1 ==y):
                return -10000
        closestGhost = min(ghostDist)       # get closest ghost to pacman (as it's the most likely danger)

        # if there's food, get manhattan distance from pacman to each food
        # if new position is same as that of a food, grant a small bonus to that move (200)
        if len(foodList) > 0:
            for food in foodList:
                x1, y1 = food
                foodDist.append(abs(x1-x) + abs(y1-y))
                if (x1 == x and y1 == y):
                    foodBonus = 200
            closestFood = min(foodDist)     # get closest food to pacman

        # grant a small penalty if pacman doesn't make a move (as to discourage staying still)
        if(newPos == currPos):
            currPosPenalty = -500
        else: currPosPenalty = 0

        if closestFood == 0:
            closestFood = 1         # so we can devide by it
        # end score is calculated as follows:
        # the closer the food, the bigger the bonus (50 / closestFood )
        # the closer the ghost, the bigger the penalty (- (200 / closestGhost))
        # if there's little food remaining, take more risks (- 100 * len(foodList)
        # if pacman doesn't change position, grant a small penalty (currPosPenalty)
        # if pacman's move leads to a food, grant a small bonus (foodBonus)
        endScore = (50/closestFood) - (200 /closestGhost) - 100 * len(foodList) + currPosPenalty + foodBonus

        return endScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, agent, depth):
            # if agent is last agent (all ghosts have been searched) to be searched then check if we are in terminal depth
            # if we are, return actions to that point and evaluate the game state
            # if not, then agent 0 moves and the depth is increased (next tree ply)
            # if agent is not last agent, then we check for terminal case and continue the minimax process
            if agent == gameState.getNumAgents():
                if depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))
                else:
                    return minimax(gameState, 0, depth + 1)
            else:
                # check if the current game state is a terminal case
                if gameState.isWin() or gameState.isLose() or depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))

                value = []          # value will be used to return the directions and the evaluation of the chosen state
                # check every action
                for action in gameState.getLegalActions(agent):
                    # get evaluation score for agent's successor's states
                    evaluation = minimax(gameState.generateSuccessor(agent, action), agent + 1, depth)[1]
                    if not value:
                        value = [action, evaluation]
                    else:
                        # if agent is pacman (0) we get the max value
                        # by replacing current value if the evaluation is higher
                        if agent == 0:
                            if value[1] < evaluation:
                                value = [action, evaluation]
                        # else we get the min value
                        # by replacing current value if the evaluation is smaller
                        else:
                            if evaluation < value[1]:
                                value = [action, evaluation]
                return value
        # begin the process at depth = 0 with agent = 0, since pacman (max player) plays first and return an action
        return minimax(gameState, 0, 0)[0]


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # process is similar to minimax process from MinimaxAgent
        # instead of determining the values inside of minimax process, we use max_value and min_value
        # processes as the project's description requires
        def minimax(gameState, agent, depth, a, b):
            if agent == gameState.getNumAgents():
                if depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))
                else:
                    return minimax(gameState, 0, depth + 1, a, b)
            else:
                # check if the current game state is a terminal case
                if gameState.isWin() or gameState.isLose() or depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))

                # if agent is pacman (0) get max_value, otherwise get min
                if agent == 0:
                    return max_value(gameState, agent, depth, a, b)
                else:
                    return min_value(gameState, agent, depth, a, b)

        def max_value(gameState, agent, depth, a, b):

            v = float("-inf")       # initialize v as minus infinity
            for action in gameState.getLegalActions(agent):
                # get evaluation score for agent's successor's states
                evaluation = minimax(gameState.generateSuccessor(agent, action), agent + 1, depth,a,b)[1]
                if v == float("-inf"):
                    v = [action, evaluation]
                else:
                    # we get max value by replacing current value with evaluation, if evaluation is higher
                    if v[1] < evaluation:
                        v = [action, evaluation]
                # if current score is higher than beta, we return v
                if v[1] > b:
                    return v
                # otherwise we get new alpha
                a = max(a, v[1])
            return v

        def min_value(gameState, agent, depth, a, b):

            v = float("inf")
            for action in gameState.getLegalActions(agent):
                evaluation = minimax(gameState.generateSuccessor(agent, action), agent + 1, depth,a,b)[1]
                if v == float("inf"):
                    v = [action, evaluation]
                else:
                    # we get min value by replacing current value with evaluation, if evaluation is smaller
                    if evaluation < v[1]:
                        v = [action, evaluation]
                # if current score is lower than alpha, we return v
                if v[1] < a:
                    return v
                # otherwise we get new beta
                b = min(b, v[1])
            return v

        return minimax(gameState,0,0,float("-inf"), float("inf"))[0]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, agent, depth):
            if agent == gameState.getNumAgents():
                if depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))
                else:
                    return minimax(gameState, 0, depth + 1)
            else:
                if gameState.isWin() or gameState.isLose() or depth == self.depth:
                    return (gameState.getLegalActions(agent), self.evaluationFunction(gameState))

                value = []
                for action in gameState.getLegalActions(agent):
                    evaluation = minimax(gameState.generateSuccessor(agent, action), agent + 1, depth)[1]
                    if not value:
                        if agent == 0:
                            value = [action, evaluation]
                        else:
                            value = [action, evaluation / len(gameState.getLegalActions(agent))]
                    else:
                        if agent == 0:
                            if value[1] < evaluation:
                                value = [action, evaluation]
                        else:
                            value = [action, value[1] + evaluation / len(gameState.getLegalActions(agent))]
                return value
        return minimax(gameState, 0, 0)[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPos = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    x, y = pos
    foodList = food.asList()
    foodDist = []       # keeps track of each food's manhattan distance from pacman
    closestFood = 0
    foodBonus = 0
    ghostDist = []      # keeps track of each ghost's manhattan distance from pacman
    endScore = 0

    # get manhattan distance from pacman to each ghost
    # if new position is same as that of a ghost return -10000 (because pacman will surely lose)
    for ghost in ghostPos:
        x1, y1 = ghost
        ghostDist.append(abs(x1-x) + abs(y1-y))
        if(x1 == x and y1 ==y):
            return -10000
    closestGhost = min(ghostDist)       # get closest ghost to pacman (as it's the most likely danger)

    # if there's food, get manhattan distance from pacman to each food
    # if new position is same as that of a food, grant a small bonus to that move (200)
    if len(foodList) > 0:
        for food in foodList:
            x1, y1 = food
            foodDist.append(abs(x1-x) + abs(y1-y))
            if (x1 == x and y1 == y):
                foodBonus = 200
        closestFood = min(foodDist)     # get closest food to pacman

    if closestFood == 0:
        closestFood = 1         # so we can devide by it
        # end score is calculated as follows:
        # the closer the food, the smaller the penalty (- closestFood)
        # the closer the ghost, the bigger the penalty (- (20 / closestGhost))
        # if there's little food remaining, take more risks (- 100 * len(foodList)
        # if pacman doesn't change position, grant a small penalty (currPosPenalty)
        # if pacman's move leads to a food, grant a small bonus (foodBonus)
    endScore = (50/closestFood) - (200 /closestGhost) - 100 * len(foodList) + foodBonus

    return endScore

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
