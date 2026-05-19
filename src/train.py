"""
- 3 escenarios exactos del lab muestreados con pesos fijos (20/30/50)
- Reward shaping con distancia geodesica (Dijkstra inflado por radio del robot)
- Observacion 13 features x 4 frames apilados (= 52 dims)
- 6 acciones discretas

Uso:
    python3 train_simple.py                       # 1.5M pasos, n_envs=cpu
    python3 train_simple.py --steps 1000000
"""
import argparse
import os
import time
from collections import deque
from heapq import heappop, heappush
from math import atan2, cos, pi, sin, sqrt

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import Robot_c, Obstacle_c, Obstacle_wall, crear_paredes_v2


# --------------------- constantes ---------------------
ARENA = 200
RADIO_OBS = 12
SENSOR_MAX = 20.0
ARENA_DIAG = ARENA * np.sqrt(2.0)
ROBOT_RADIUS = 5.0
GRID_CELL = 5
GRID_N = ARENA // GRID_CELL

ACTION_TABLE = np.array([
    [ 1.0,  1.0], [ 0.5,  0.5], [-0.6,  0.6],
    [ 0.6, -0.6], [ 0.5,  1.0], [ 1.0,  0.5],
], dtype=np.float32)

FRAME_STACK = 4
OBS_FEATURES = 13
OBS_DIM = FRAME_STACK * OBS_FEATURES


# --------------------- escenarios ---------------------
def _wall(traj):
    pts = crear_paredes_v2(traj)
    return [Obstacle_wall(p[0], p[1], ARENA, 1.25, i, len(pts)) for i, p in enumerate(pts)]


def _perimeter():
    return _wall([[1, 1], [199, 1], [199, 199], [1, 199], [1, 1]])


def scenario_1(rng):
    obj = [185, 185, 10]
    rx = 25.0 + float(rng.normal(0, 4))
    ry = 10.0 + float(rng.normal(0, 4))
    rt = pi / 4 + float(rng.normal(0, 0.3))
    robot = Robot_c(rx, ry, rt, obj)
    obstacles = [Obstacle_c(100, 100, ARENA, RADIO_OBS, 0, 1)] + _perimeter()
    return robot, obstacles, obj


def scenario_2(rng):
    pos_x = float(rng.uniform(0.1, 0.9) * ARENA)
    obj = [pos_x, 185, 10]
    rt = pi / 4 + float(rng.normal(0, 0.3))
    robot = Robot_c(ARENA - pos_x, 10, rt, obj)
    obstacles = [Obstacle_c(-1, -1, ARENA, RADIO_OBS, 0, 1)] + _perimeter()
    obstacles += _wall([[200, 50], [125, 125], [75, 75]])
    return robot, obstacles, obj


def scenario_3(rng):
    pos_x = float(rng.uniform(0.1, 0.9) * ARENA)
    obj = [pos_x, 185, 10]
    rt = pi / 4 + float(rng.normal(0, 0.3))
    robot = Robot_c(ARENA - pos_x, 10, rt, obj)
    obstacles = [Obstacle_c(-1, -1, ARENA, RADIO_OBS, i, 3) for i in range(3)]
    obstacles += _perimeter()
    obstacles += _wall([[50, 0], [75, 75], [50, 150]])
    obstacles += _wall([[150, 150], [150, 200]])
    return robot, obstacles, obj


SCENARIOS = [scenario_1, scenario_2, scenario_3]

# Curriculum minimo: 3 fases segun fraccion del entrenamiento.
WARMUP = [
    (0.15, [1.00, 0.00, 0.00]),  # 0-15% : solo scen1 (lo facil primero)
    (0.40, [0.30, 0.60, 0.10]),  # 15-40%: scen2 dominante, le da tiempo a consolidar
    (1.00, [0.20, 0.40, 0.40]),  # 40-100%: scen2 y scen3 al mismo peso (anti-olvido)
]


# --------------------- geodesica ---------------------
def compute_geodesic(obstacles, objetivo):
    blocked = np.zeros((GRID_N, GRID_N), dtype=bool)
    for ob in obstacles:
        eff_r = ob.radius + ROBOT_RADIUS
        rc = int(np.ceil((eff_r + GRID_CELL / 2) / GRID_CELL))
        ci, cj = int(ob.x // GRID_CELL), int(ob.y // GRID_CELL)
        for di in range(-rc, rc + 1):
            for dj in range(-rc, rc + 1):
                i, j = ci + di, cj + dj
                if 0 <= i < GRID_N and 0 <= j < GRID_N:
                    cx, cy = (i + 0.5) * GRID_CELL, (j + 0.5) * GRID_CELL
                    if sqrt((cx - ob.x) ** 2 + (cy - ob.y) ** 2) <= eff_r + GRID_CELL / 2:
                        blocked[i, j] = True

    gi = min(GRID_N - 1, max(0, int(objetivo[0] // GRID_CELL)))
    gj = min(GRID_N - 1, max(0, int(objetivo[1] // GRID_CELL)))
    blocked[gi, gj] = False

    dist = np.full((GRID_N, GRID_N), GRID_N * 2.0, dtype=np.float32)
    dist[gi, gj] = 0.0
    pq = [(0.0, gi, gj)]
    SQ2 = sqrt(2.0)
    while pq:
        d, i, j = heappop(pq)
        if d > dist[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if not (0 <= ni < GRID_N and 0 <= nj < GRID_N):
                    continue
                if blocked[ni, nj]:
                    continue
                nd = d + (SQ2 if (di and dj) else 1.0)
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heappush(pq, (nd, ni, nj))
    return dist


# --------------------- env ---------------------
class RobotNavEnv(gym.Env):
    def __init__(self, seed=None, max_steps=800):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_DIM,), dtype=np.float32)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.stack = deque(maxlen=FRAME_STACK)
        self.weights = list(WARMUP[0][1])  # arranca con la fase 0
        self._reset_internal()

    def set_weights(self, w):
        self.weights = list(w)

    def _reset_internal(self):
        s = int(self.rng.choice(len(SCENARIOS), p=self.weights))
        self.robot, self.obstacles, self.objetivo = SCENARIOS[s](self.rng)
        self.robot.updatePosition(0.0, 0.0)
        for ob in self.obstacles:
            self.robot.updateSensors(ob)
        self.dist_grid = compute_geodesic(self.obstacles, self.objetivo)
        self.prev_geo = self._geo()
        self.t = 0
        obs0 = self._obs()
        self.stack.clear()
        for _ in range(FRAME_STACK):
            self.stack.append(obs0)

    def _geo(self):
        i = min(GRID_N - 1, max(0, int(self.robot.x // GRID_CELL)))
        j = min(GRID_N - 1, max(0, int(self.robot.y // GRID_CELL)))
        return float(self.dist_grid[i, j])

    def _obs(self):
        readings = np.array(
            [s.reading if s.reading > 0 else SENSOR_MAX for s in self.robot.prox_sensors],
            dtype=np.float32) / SENSOR_MAX
        dx = self.objetivo[0] - self.robot.x
        dy = self.objetivo[1] - self.robot.y
        rel = (atan2(dy, dx) - self.robot.theta + pi) % (2 * pi) - pi
        return np.array([
            *readings, sin(rel), cos(rel),
            sqrt(dx * dx + dy * dy) / ARENA_DIAG, self.robot.vl, self.robot.vr,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return np.concatenate(self.stack), {}

    def step(self, action):
        vl, vr = ACTION_TABLE[int(action)]
        self.robot.updatePosition(vl, vr)
        collision = False
        for ob in self.obstacles:
            self.robot.collisionCheck(ob)
            self.robot.updateSensors(ob)
            if self.robot.stall == 1:
                collision = True
        obs = self._obs()
        self.stack.append(obs)

        cur_geo = self._geo()
        r = (self.prev_geo - cur_geo) - 0.05 - 0.05 * abs(vr - vl) + 0.02 * (vl + vr) / 2
        if float(obs[:8].min()) * SENSOR_MAX < 4.0:
            r -= 0.1
        self.prev_geo = cur_geo

        dx = self.objetivo[0] - self.robot.x
        dy = self.objetivo[1] - self.robot.y
        done = False
        if collision:
            r -= 10.0
            done = True
        elif sqrt(dx * dx + dy * dy) < self.objetivo[2]:
            r += 100.0
            done = True

        self.t += 1
        return np.concatenate(self.stack), float(r), done, self.t >= self.max_steps, {}


def make_env(rank, seed):
    def _init():
        np.random.seed(seed + rank * 997)  # semilla global por obstaculos random
        return RobotNavEnv(seed=seed + rank * 997)
    return _init


# --------------------- main ---------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_500_000)
    p.add_argument('--n-envs', type=int, default=0, help='0 = cpu_count')
    p.add_argument('--output', type=str, default='dqn_robot.zip')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    class WarmupCB(BaseCallback):
        def __init__(self, total):
            super().__init__()
            self.total = total
            self.phase = -1

        def _on_step(self) -> bool:
            frac = self.num_timesteps / self.total
            for i, (lim, w) in enumerate(WARMUP):
                if frac < lim:
                    if i != self.phase:
                        self.phase = i
                        self.training_env.env_method('set_weights', w)
                        print(f'[curriculum] fase {i}  weights={w}  step={self.num_timesteps}')
                    break
            return True

    n_envs = args.n_envs if args.n_envs > 0 else max(1, os.cpu_count() or 1)
    torch.set_num_threads(1)
    print(f'[setup] n_envs={n_envs}  steps={args.steps}')

    if n_envs == 1:
        env = DummyVecEnv([make_env(0, args.seed)])
    else:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(n_envs)])

    model = DQN(
        'MlpPolicy', env,
        learning_rate=5e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=128,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=WarmupCB(args.steps),
                progress_bar=False)
    print(f'[train] fin ({(time.time() - t0) / 60:.1f} min)')

    model.save(args.output)
    print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
