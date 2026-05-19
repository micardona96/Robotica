# Robótica

Entrenamiento DQN para el robot `dqn_robot.zip`

## DQN

Entrena una red Q sobre 3 escenarios fijos (los del lab):

1. **scen1**: arena con un obstáculo central. Inicio y meta deterministas.
2. **scen2**: arena con un muro diagonal + un obstáculo aleatorio.
3. **scen3**: arena con dos muros + tres obstáculos aleatorios.

El episodio termina cuando el robot toca la meta (+100) o choca (-10).
Cada escenario se muestrea por episodio con pesos que dependen de la
fase del curriculum.

## Cómo lo hace

### Acciones (discretas)

Seis acciones precomputadas en `ACTION_TABLE`: avanzar rápido/lento,
girar izquierda/derecha en el sitio, curva suave izquierda/derecha.
Cada acción es un par `(vl, vr)` directo, sin escalas.

### Observación

13 features por frame, 4 frames apilados → vector de 52 dims:

- 8 lecturas de sensores normalizadas a [0,1] (1 = libre, 0 = pegado)
- `sin` y `cos` del ángulo relativo a la meta
- distancia euclidiana normalizada por la diagonal de la arena
- `vl` y `vr` actuales (memoria de la acción anterior)

El frame stacking se hace dentro del entorno con una `deque(maxlen=4)`.
Rompe la simetría que causaría oscilación con observación puntual.

### Reward shaping con distancia geodésica

Antes de cada episodio se calcula con Dijkstra una grilla 40x40 de
distancias reales al objetivo, inflando los obstáculos por el radio
del robot para no proponer rutas físicamente imposibles. El reward
por paso es:

    progreso_geodésico − 0.05 − 0.05·|vr−vl| + 0.02·(vl+vr)/2 − 0.1 [si sensor<4]

El progreso es la diferencia entre la distancia geodésica del paso
anterior y el actual. Si el agente rodea un obstáculo, gana reward
porque se acerca por el camino real, no en línea recta. Sin esto,
en scen3 el agente queda atrapado oscilando contra paredes.

### Curriculum mínimo (`WARMUP`)

Tres fases que cambian los pesos de muestreo según el avance:

| Fase     | Pesos [s1, s2, s3] | Razón                              |
|----------|--------------------|------------------------------------|
| 0–15%    | [1.00, 0.00, 0.00] | aprende lo básico en arena vacía   |
| 15–40%   | [0.30, 0.60, 0.10] | consolida scen2 antes de scen3     |
| 40–100%  | [0.20, 0.40, 0.40] | mezcla final, anti-olvido balanceado |

Sin la fase 0, todas las primeras transiciones de scen3 son choques
con política random, la red aprende `Q(scen3, *) ≈ −10` y nunca sale
del óptimo local. La fase intermedia evita catastrophic forgetting
de scen2 cuando llega scen3.

Las fases las controla un `BaseCallback` (`WarmupCB`) que llama
`env.env_method('set_weights', ...)` cuando cruza un umbral.

### Paralelismo

`SubprocVecEnv` con `n_envs = os.cpu_count()` lanza un proceso por
core. Cada worker tiene su semilla global `np.random.seed(seed + rank*997)`
para que los obstáculos aleatorios de `Obstacle_c(-1, -1, ...)` sean
reproducibles. `torch.set_num_threads(1)` evita contención del GIL.


## Uso

```bash
python3 train.py                       # 1.5M pasos, n_envs=cpu_count
python3 train.py --steps 3000000       # más pasos = mejor scen3
```
