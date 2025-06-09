import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Define constants
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200

ACTION_2_MS = 12.5

ALT_DIF_REWARD_SCALE = -5/3000
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -50/3000

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

AC_SPD = 150

ACTION_FREQUENCY = 30

class DescentEnvNoDtype(gym.Env):
    """
    Igual que DescentEnv pero sin usar dtype en los spaces.
    """

    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 256
        self.window_size = (self.window_width, self.window_height)

        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "target_altitude": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "runway_distance": spaces.Box(-np.inf, np.inf, shape=(1,))
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        self.total_reward = 0
        self.final_altitude = 0

        self.window = None
        self.clock = None

    def _get_obs(self):
        DEFAULT_RWY_DIS = 200 
        RWY_LAT = 52
        RWY_LON = 4
        NM2KM = 1.852

        self.altitude = bs.traf.alt[0]
        self.vz = bs.traf.vs[0]
        self.runway_distance = (DEFAULT_RWY_DIS - bs.tools.geo.kwikdist(RWY_LAT,RWY_LON,bs.traf.lat[0],bs.traf.lon[0])*NM2KM)

        obs_altitude = np.array([(self.altitude - ALT_MEAN)/ALT_STD])
        obs_vz = np.array([(self.vz - VZ_MEAN) / VZ_STD])
        obs_target_alt = np.array([((self.target_alt- ALT_MEAN)/ALT_STD)])
        obs_runway_distance = np.array([(self.runway_distance - RWY_DIS_MEAN)/RWY_DIS_STD])

        observation = {
                "altitude": obs_altitude,
                "vz": obs_vz,
                "target_altitude": obs_target_alt,
                "runway_distance": obs_runway_distance,
            }
        
        return observation

    def _get_info(self):
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }

    def _get_reward(self):
        if self.runway_distance > 0 and self.altitude > 0:
            reward = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
            self.total_reward += reward
            return reward, 0
        elif self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = -100
            self.total_reward += reward
            return reward, 1
        elif self.runway_distance <= 0:
            reward = self.altitude * RWY_ALT_DIF_REWARD_SCALE
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, 1

    def _get_action(self,action):
        action = action * ACTION_2_MS
        if action >= 0:
            bs.traf.selalt[0] = 1000000
            bs.traf.selvs[0] = action
        elif action < 0:
            bs.traf.selalt[0] = 0
            bs.traf.selvs[0] = action

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_reward = 0
        self.final_altitude = 0

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF,TARGET_ALT_DIF)

        bs.traf.cre('KL001',actype="A320",acalt=alt_init,acspd=AC_SPD)
        bs.traf.swvnav[0] = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._get_action(action)
        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
                observation = self._get_obs()

        observation = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()

        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        zero_offset = 25
        max_distance = 180

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        pygame.draw.rect(
            canvas, 
            (154,205,50),
            pygame.Rect(
                (0,self.window_height-50),
                (self.window_width, 50)
                ),
        )
        
        max_alt = 5000
        target_alt = int((-1*(self.target_alt-max_alt)/max_alt)*(self.window_height-50))

        pygame.draw.line(
            canvas,
            (255,255,255),
            (0,target_alt),
            (self.window_width,target_alt)
        )

        runway_length = 30
        runway_start = int(((self.runway_distance + zero_offset)/max_distance)*self.window_width)
        runway_end = int(runway_start + (runway_length/max_distance)*self.window_width)

        pygame.draw.line(
            canvas,
            (119,136,153),
            (runway_start,self.window_height - 50),
            (runway_end,self.window_height - 50),
            width = 3
        )

        aircraft_alt = int((-1*(self.altitude-max_alt)/max_alt)*(self.window_height-50))
        aircraft_start = int(((zero_offset)/max_distance)*self.window_width)
        aircraft_end = int(aircraft_start + (4/max_distance)*self.window_width)

        pygame.draw.line(
            canvas,
            (0,0,0),
            (aircraft_start,aircraft_alt),
            (aircraft_end,aircraft_alt),
            width = 5
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass