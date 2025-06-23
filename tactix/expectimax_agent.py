from agent import Agent
import numpy as np

class ExpectimaxTacTixAgent(Agent):
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

    def pol_prob(self, board, action):
        valid_actions = self.get_valid_actions(board)
        return 1 / len(valid_actions) if action in valid_actions else 0 # probabilidad uniforme de elegir una acción válida
    
    def h1(self, board):
        # Heurística simple: paridad de piezas restantes
        remaining = np.count_nonzero(board)
        return 1 if remaining % 2 == 1 else -1

    def h2(self, board):
        # Heurística de segmentos ponderados: evalúa la calidad de los segmentos disponibles
        score = 0
        size = board.shape[0]
        
        # Evaluar filas y columnas
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                # Encontrar segmentos consecutivos
                segments = []
                start = None
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        segments.append(i - start)  # longitud del segmento
                        start = None
                if start is not None:
                    segments.append(size - start)
                
                # Asignar puntuación basada en el tamaño de los segmentos
                for seg_len in segments:
                    if seg_len == 1:
                        score += 1    # Segmentos pequeños son menos valiosos
                    elif seg_len == 2:
                        score += 3    # Segmentos medianos
                    elif seg_len == 3:
                        score += 5    # Segmentos grandes
                    else:
                        score += 7    # Segmentos muy grandes son muy valiosos
        
        # Aplicar paridad como factor de corrección
        remaining = np.count_nonzero(board)
        parity_factor = 1 if remaining % 2 == 1 else -1 # para quien va
        
        return score * parity_factor
    
    def h(self, board):
        return self.h1(board) + 2 * self.h2(board) # ponderación de heurísticas, es más importante la segunda

    def expectimax(self, board, depth, maximizing_player):
        if depth == 0 or np.count_nonzero(board) == 0:
            return self.h(board)
        
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return 0

        if maximizing_player: # jugador maximizador
            values = []
            for action in valid_actions:
                new_board = self.simulate_action(board, action)
                val = self.expectimax(new_board, depth - 1, False)
                values.append(val)
            return max(values)
        else: # jugador minimizador (expectativa)
            expected_value = 0
            for action in valid_actions:
                new_board = self.simulate_action(board, action)
                prob = self.pol_prob(board, action)
                val = self.expectimax(new_board, depth-1, True)
                expected_value += prob * val
            return expected_value

    def act(self, observation):
        board = observation["board"]
        valid_actions = self.get_valid_actions(board)
        best_action = None
        best_value = float('-inf')
        for action in valid_actions:
            new_board = self.simulate_action(board, action)
            value = self.expectimax(new_board, self.depth - 1, False)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
