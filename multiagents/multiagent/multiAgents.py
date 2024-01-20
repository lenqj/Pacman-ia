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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Lista de poziții ale mancarii
        foodList = newFood.asList()

        # Lista de poziții ale fantomelor
        ghostPos = [ghost.getPosition() for ghost in newGhostStates]

        score = 0

        # Calculează distanțele până la hrana și fantomele în bucle
        for i in foodList:
            # Calculează distanța până la hrana curentă
            foodDistance = abs(i[0] - newPos[0]) + abs(i[1] - newPos[1])

            # Ajustează scorul în funcție de distanța până la hrana
            if foodDistance == 0:
                score += 10
            elif foodDistance == 1:
                score += 5
            elif foodDistance == 2:
                score += 3
            elif foodDistance == 3:
                score += 2
        for i in ghostPos:
            # Calculează distanța până la fantoma curentă
            ghostDistance = abs(i[0] - newPos[0]) + abs(i[1] - newPos[1])
            # Ajustează scorul în funcție de distanța până la fantomă
            if ghostDistance < 4:
                if ghostDistance == 1:
                    score -= 500
                else:
                    score -= 10
        # Verifică dacă acțiunea este o oprire și penalizează
        if action == Directions.STOP:
            score -= 10
        # Dacă jocul este câștigat, returnează un scor mare
        if successorGameState.isWin():
            return 1000
        # Adaugă diferența de scor
        score += successorGameState.getScore() - currentGameState.getScore()

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


from game import Directions, random


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        legalActions = gameState.getLegalActions(0)  # Actiuni legale
        scores = [self.mini(gameState.generateSuccessor(0, action), self.depth, 1) for action in legalActions]

        # Alege acțiunea cu scorul maxim
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) if bestIndices else 0  # Default la prima acțiune
        return legalActions[chosenIndex]

    def maxi(self, state, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        v = float('-inf')
        for action in state.getLegalActions(0):  # Acțiunile lui Pacman
            successor = state.generateSuccessor(0, action)
            v = max(v, self.mini(successor, depth, 1))
        return v

    def mini(self, state, depth, ghostIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        v = float('inf')
        for action in state.getLegalActions(ghostIndex):
            successor = state.generateSuccessor(ghostIndex, action)

            # Dacă există mai mulți fantome, apelează min_value cu următorul index de fantomă
            if ghostIndex < state.getNumAgents() - 1:
                v = min(v, self.mini(successor, depth, ghostIndex + 1))
            else:
                v = min(v, self.maxi(successor, depth - 1))  # Tura lui Pacman

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def terminalState(state, adancime, apel_max):
            if apel_max == 1:
                if state.isWin() or state.isLose() or adancime == self.depth:
                    return True
                else:
                    return False
            else:
                if state.isWin() or state.isLose():
                    return True
                else:
                    return False

        # folosim pentru pacman deci AgentIndex va fi 0 mereu
        def maxScor(state, adancime, alpha, beta):
            if terminalState(state, adancime, 1):
                return self.evaluationFunction(state), ''
            else:
                scor = -100000
                mutare = ''
                alpha_curent = alpha
                for action in state.getLegalActions(0):
                    scor_temporar, nimic = minScor(state.generateSuccessor(0, action), 1, adancime, alpha_curent, beta)
                    if scor_temporar > scor:
                        scor = scor_temporar
                        mutare = action
                        if scor > alpha_curent:
                            alpha_curent = scor
                    if scor > beta:
                        return scor, mutare
                return scor, mutare

        # folosim pentru fantome deci vom avea AgentIndex >= 1
        def minScor(state, index_agent, adancime, alpha, beta):
            if terminalState(state, adancime, 0):
                return self.evaluationFunction(state), ''
            else:
                scor = 100000
                mutare = ''
                beta_curent = beta
                for action in state.getLegalActions(index_agent):
                    scor_temporar = 0
                    if index_agent == (state.getNumAgents() - 1):
                        scor_temporar, nimic = maxScor(state.generateSuccessor(index_agent, action), adancime + 1,
                                                       alpha, beta_curent)
                    else:
                        scor_temporar, nimic = minScor(state.generateSuccessor(index_agent, action), index_agent + 1,
                                                       adancime, alpha, beta_curent)
                    if scor_temporar < scor:
                        scor = scor_temporar
                        mutare = action
                        if beta_curent > scor:
                            beta_curent = scor
                    if scor < alpha:
                        return scor, mutare
                return scor, mutare

        # cel mai de sus apel al algoritmului
        scor, final_action = maxScor(gameState, 0, -100000, 100000)
        return final_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        "* YOUR CODE HERE *"

        def expectimax(gameState, agentIndex, depth=0):
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None
            # If terminal(pos)
            if (gameState.isLose() or gameState.isWin() or depth == self.depth):
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1

            numAction = len(legalActionList)
            # if player(pos) == MAX: value = -infinity
            if agentIndex == self.index:
                value = -float("inf")
            # if player(pos) == CHANCE: value = 0
            else:
                value = 0

            for legalAction in legalActionList:
                successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                expectedMax = expectimax(successorGameState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        value = expectedMax
                        bestAction = legalAction
                else:
                    value = value + ((1.0 / numAction) * expectedMax)
            return value, bestAction

        bestScoreActionPair = expectimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove = bestScoreActionPair[1]
        return bestMove




def betterEvaluationFunction(currentGamState: GameState):
    pozitiaPacman = currentGamState.getPacmanPosition()
    pozitiiMancare = currentGamState.getFood().asList()
    stariFantomelor = currentGamState.getGhostStates()

    pondereMancare = 10
    pondereFantome = -100
    pondereDistanta = -1
    # Calculeaza scorul bazat pe diferiti factori
    scor = currentGamState.getScore()
    # Evalueaza pozitiile mancarii
    distanteMancare = [manhattanDistance(pozitiaPacman, mancare) for mancare in pozitiiMancare]
    if distanteMancare:
        ceaMaiApropiataDistantaMancare = min(distanteMancare)
        scor += pondereMancare / ceaMaiApropiataDistantaMancare
    # Evalueaza pozitiile fantomelor
    for stareFantoma in stariFantomelor:
        pozitieFantoma = stareFantoma.getPosition()
        distantaFantoma = manhattanDistance(pozitiaPacman, pozitieFantoma)
        if distantaFantoma < 2:
            scor += pondereFantome
    return scor
# Restul codului
better = betterEvaluationFunction

