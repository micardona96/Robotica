"""
Carga dqn_robot.zip (entrenado con train.py) y devuelve (vl, vr) en cada paso.
"""
import numpy as np
from collections import deque
from math import atan2, cos, pi, sin, sqrt
from stable_baselines3 import DQN

SENSOR_MAX = 20.0
ARENA_DIAG = 200.0 * np.sqrt(2.0)
FRAME_STACK = 4

ACTION_TABLE = np.array([
    [ 1.0,  1.0], [ 0.5,  0.5], [-0.6,  0.6],
    [ 0.6, -0.6], [ 0.5,  1.0], [ 1.0,  0.5],
], dtype=np.float32)


class Controller_c:
    def __init__(self, model_path='dqn_robot.zip'):
        self.model = DQN.load(model_path, device='cpu')
        self.stack = deque(maxlen=FRAME_STACK)

    def update(self, robot, objetivo):
        # Si ya llego a la meta, no hacer nada (desire==1 lo marca el simulador).
        if robot.desire == 1:
            return 0.0, 0.0
        readings = np.array(
            [s.reading if s.reading > 0 else SENSOR_MAX for s in robot.prox_sensors],
            dtype=np.float32) / SENSOR_MAX
        dx, dy = objetivo[0] - robot.x, objetivo[1] - robot.y
        rel = (atan2(dy, dx) - robot.theta + pi) % (2 * pi) - pi
        obs = np.array([
            *readings, sin(rel), cos(rel),
            sqrt(dx * dx + dy * dy) / ARENA_DIAG, robot.vl, robot.vr,
        ], dtype=np.float32)

        while len(self.stack) < FRAME_STACK:
            self.stack.append(obs)
        self.stack.append(obs)

        action, _ = self.model.predict(np.concatenate(self.stack), deterministic=True)
        vl, vr = ACTION_TABLE[int(action)]
        return float(vl), float(vr)
