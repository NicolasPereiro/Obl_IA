from agent import Agent
import numpy as np

class MinimaxTacTixAgent(Agent):
    def __init__(self, env, depth=4):
        self.env = env
        self.depth = depth
        
    def get_valid_actions(self, board):
        actions = []
        size = board.shape[0]
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        actions.append([idx, start, i-1, is_row])
                        start = None
                if start is not None:
                    actions.append([idx, start, size-1, is_row])
        return actions
    
    def simulate_action(self, board, action):
        # Simula el resultado de aplicar una acción sobre una copia del tablero
        idx, start, end, is_row = action
        new_board = board.copy()
        if is_row:
            new_board[idx, start:end+1] = 0
        else:
            new_board[start:end+1, idx] = 0
        return new_board
    
    def h(self, board):
        # Heurística simple: paridad de piezas restantes
        remaining = np.count_nonzero(board)
        return 1 if remaining % 2 == 1 else -1


    def minimax(self, board, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or np.count_nonzero(board) == 0:
            return self.h(board)
        
        valid_actions = self.get_valid_actions(board)
        if maximizing_player:
            max_eval = float('-inf')
            for action in valid_actions:
                new_board = self.simulate_action(board, action)
                eval = self.minimax(new_board, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Poda beta
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                new_board = self.simulate_action(board, action)
                eval = self.minimax(new_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Poda alfa
            return min_eval

    def act(self, observation):
        board = observation["board"]
        valid_actions = self.get_valid_actions(board)
        best_action = None
        best_value = float('-inf')
        for action in valid_actions:
            new_board = self.simulate_action(board, action)
            value = self.minimax(new_board, self.depth - 1, False)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
