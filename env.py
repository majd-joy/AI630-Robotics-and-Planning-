"""
env.py - Grid-world robot environment modelled as an MDP (Tasks 1 and 2).

State  : a cell (row, col) of the grid, encoded as s = row * W + col
Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
Reward : -1 per movement, +100 on reaching the goal, -10 for trying to enter
         an obstacle / leave the grid (the robot then stays in place)
Slip   : with probability `slip` the robot moves perpendicular to the intended
         direction (slip/2 for each side). slip = 0 -> deterministic world,
         slip = 0.2 -> "moves as intended with prob. 0.8" (Experiment C).

The full transition model P(s'|s,a) and reward R(s,a,s') are stored in
`env.P[(s, a)] = [(prob, next_state, reward, done, collided), ...]`
so Dynamic Programming can use them, while model-free methods only call
`env.step(...)` (interaction with the environment).
"""

import random

ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ARROWS = ["\u2191", "\u2193", "\u2190", "\u2192"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPENDICULAR = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

# '#' obstacle, 'S' start (upper-left), 'G' goal (lower-right), '.' free cell
SIMPLE_MAZE = [
    "S.......",
    ".##.....",
    ".....#..",
    "..#..#..",
    "..#.....",
    ".....#..",
    ".##..#..",
    ".......G",
]

COMPLEX_MAZE = [
    "S..#....",
    ".#.#.##.",
    ".#...#..",
    ".####.#.",
    "......#.",
    ".####.#.",
    "...#....",
    ".#...##G",
]


def generate_random_maze(height=8, width=8, obstacle_probability=0.20,
                         rng=None, max_attempts=1000):
    """
    Generate a random, solvable maze.

    The start is fixed at the upper-left corner and the goal at the
    lower-right corner. Every other cell independently becomes an obstacle
    with probability `obstacle_probability`.

    Returns a list of strings in the same format as SIMPLE_MAZE/COMPLEX_MAZE.
    A ValueError is raised if a solvable maze cannot be generated.
    """
    if height < 2 or width < 2:
        raise ValueError("height and width must both be at least 2")
    if not 0.0 <= obstacle_probability < 1.0:
        raise ValueError("obstacle_probability must be in [0, 1)")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    rng = rng if rng is not None else random.Random()

    for _ in range(max_attempts):
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                is_endpoint = (r == 0 and c == 0) or (
                    r == height - 1 and c == width - 1
                )
                row.append("#" if (not is_endpoint and
                                   rng.random() < obstacle_probability) else ".")
            grid.append(row)

        grid[0][0] = "S"
        grid[height - 1][width - 1] = "G"
        layout = ["".join(row) for row in grid]

        # Check reachability without depending on a GridWorld instance.
        start, goal = 0, height * width - 1
        obstacles = {
            r * width + c
            for r, row in enumerate(layout)
            for c, cell in enumerate(row)
            if cell == "#"
        }
        seen = {start}
        queue = [start]
        head = 0
        while head < len(queue):
            s = queue[head]
            head += 1
            r, c = divmod(s, width)
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    nxt = nr * width + nc
                    if nxt not in obstacles and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)

        if goal in seen:
            return layout

    raise ValueError(
        "Could not generate a solvable maze. Try a lower obstacle_probability "
        "or a larger max_attempts."
    )



class GridWorld:
    @classmethod
    def generate_random(cls, height=8, width=8, obstacle_probability=0.20,
                        slip=0.0, step_reward=-1.0, goal_reward=100.0,
                        collision_penalty=-10.0, seed=None, max_attempts=1000):
        """Create a GridWorld using a newly generated solvable maze."""
        rng = random.Random(seed)
        layout = generate_random_maze(
            height=height,
            width=width,
            obstacle_probability=obstacle_probability,
            rng=rng,
            max_attempts=max_attempts,
        )
        return cls(layout, slip, step_reward, goal_reward, collision_penalty)

    def __init__(self, layout=SIMPLE_MAZE, slip=0.0, step_reward=-1.0,
                 goal_reward=100.0, collision_penalty=-10.0):
        self.layout = layout
        self.H, self.W = len(layout), len(layout[0])
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.n_states = self.H * self.W
        self.n_actions = 4

        self.obstacles = set()
        for r, row in enumerate(layout):
            for c, ch in enumerate(row):
                s = r * self.W + c
                if ch == "#":
                    self.obstacles.add(s)
                elif ch == "S":
                    self.start = s
                elif ch == "G":
                    self.goal = s
        self.free_states = [s for s in range(self.n_states) if s not in self.obstacles]
        self.decision_states = [s for s in self.free_states if s != self.goal]
        self._build_model()

    # ---------------------------------------------------------------- helpers
    def rc(self, s):
        return divmod(s, self.W)

    def _move(self, s, a):
        """Deterministic effect of trying to move in direction a from s."""
        r, c = self.rc(s)
        dr, dc = MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W) or (nr * self.W + nc) in self.obstacles:
            return s, self.collision_penalty, False, True      # collision: stay in place
        s2 = nr * self.W + nc
        if s2 == self.goal:
            return s2, self.goal_reward, True, False
        return s2, self.step_reward, False, False

    def _build_model(self):
        """Transition model P(s'|s,a) and reward R(s,a,s') (known to planners)."""
        self.P = {}
        for s in self.free_states:
            for a in range(self.n_actions):
                if s == self.goal:                         # terminal state
                    self.P[(s, a)] = [(1.0, s, 0.0, True, False)]
                    continue
                if self.slip == 0:
                    outcomes = [(1.0, a)]
                else:
                    p1, p2 = PERPENDICULAR[a]
                    outcomes = [(1 - self.slip, a), (self.slip / 2, p1), (self.slip / 2, p2)]
                self.P[(s, a)] = [(p,) + self._move(s, b) for p, b in outcomes]

    # ------------------------------------------------------------- interaction
    def step(self, s, a, rng):
        """Sample one real transition. Returns (next_state, reward, done, collided)."""
        u, acc = rng.random(), 0.0
        outcomes = self.P[(s, a)]
        for p, s2, r, done, col in outcomes:
            acc += p
            if u <= acc:
                return s2, r, done, int(col)
        p, s2, r, done, col = outcomes[-1]
        return s2, r, done, int(col)

    def is_connected(self):
        """Return True if the goal is reachable from the start (BFS)."""
        seen = {self.start}
        queue = [self.start]
        head = 0
        while head < len(queue):
            s = queue[head]
            head += 1
            for a in range(self.n_actions):
                s2 = self._move(s, a)[0]
                if s2 not in seen:
                    seen.add(s2)
                    queue.append(s2)
        return self.goal in seen


if __name__ == "__main__":
    # The two predefined mazes are examples for controlled comparisons.
    for name, lay in [("simple example", SIMPLE_MAZE),
                      ("complex example", COMPLEX_MAZE)]:
        env = GridWorld(lay)
        print(name, "free cells:", len(env.free_states),
              "reachable goal:", env.is_connected())

    # A fresh random maze can be generated for each experiment.
    random_env = GridWorld.generate_random(
        height=8, width=8, obstacle_probability=0.20, seed=42
    )
    print("random maze", "free cells:", len(random_env.free_states),
          "reachable goal:", random_env.is_connected())
    print("\n".join(random_env.layout))
