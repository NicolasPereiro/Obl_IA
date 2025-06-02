import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TacTixEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, board_size=6, misere=False, render_mode="human"):
        super().__init__()
        self.board_size = board_size
        self.misere = misere
        self.render_mode = render_mode
        self.board = np.ones((board_size, board_size), dtype=np.int32)
        self.done = False
        self.current_player = 0  # 0 or 1

        self.action_space = spaces.MultiDiscrete([board_size, board_size, board_size, 2])
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int32),
            "current_player": spaces.Discrete(2)
        })

    def reset(self):
        self.board = np.ones((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "current_player": self.current_player
        }

    def _valid_action(self, idx, start, end, is_row):
        if not (0 <= idx < self.board_size and 0 <= start <= end < self.board_size):
            return False
        if is_row:
            return np.all(self.board[idx, start:end+1] == 1)
        else:
            return np.all(self.board[start:end+1, idx] == 1)

    def step(self, action):
        idx, start, end, is_row = action
        is_row = bool(is_row)

        if not self._valid_action(idx, start, end, is_row):
            raise ValueError("Invalid action.")

        if is_row:
            self.board[idx, start:end+1] = 0
        else:
            self.board[start:end+1, idx] = 0

        if np.count_nonzero(self.board) == 0:
            self.done = True
            reward = -1 if self.misere else 1
        else:
            reward = 0

        self.current_player = 1 - self.current_player
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        board_str = "\n".join(' '.join('O' if cell else '.' for cell in row) for row in self.board)

        if self.done:
            last_player = 1 - self.current_player
            if self.misere:
                winner = self.current_player + 1
            else:
                winner = last_player + 1
            board_str += f"\n\n🎉 Player {winner} wins! ({'Misère' if self.misere else 'Normal'} rules)"
        else:
            board_str += f"\nPlayer {self.current_player + 1}'s turn ({'Misère' if self.misere else 'Normal'} rules)\n"

        if self.render_mode == "human":
            print(board_str)
        elif self.render_mode == "rgb_array":
            return self._text_to_image(board_str)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not supported.")

    def _text_to_image(self, text):
        font = ImageFont.load_default()
        lines = text.split("\n")

        width, height = 640, 480

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        widths = []
        line_height = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            widths.append(width)
            line_height = max(line_height, height)

        width = max(widths) + 20
        height = line_height * len(lines) + 20

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        for i, line in enumerate(lines):
            draw.text((10, 10 + i * line_height), line, fill="black", font=font)

        return np.array(image)
