from .graph import CityGraph
import random
from termcolor import colored


class City:
    # Set reward/punishment values
    NORMAL_REW = 1
    MEDIUM_REW = 5
    SEVERE_REW = 10

    # Set max number of tasks (1-indexed)
    MAX_TASKS = 10

    def __init__(self, spawn_point=(0, 0)):
        """
        City with 6 streets and 4 avenues, where streets cut horizontally and are one-way, while avenues cut vertically and are two-way

        Cell reference:
        - 0: building
        - 1: road
        - P: player
        - A/S: traffic light (A is stop avenue, S is stop street)
        """

        num_rows, num_cols = 16, 12
        building_ys = (1, 2, 5, 6, 9, 10)

        assert (
            0 <= spawn_point[0] < num_rows and 0 <= spawn_point[1] < num_cols
        ), "Player cannot spawn off-grid; grid has 16 rows, 12 cols."

        # Init roads and buildings
        self.grid = [
            (
                [0 if i in building_ys else 1 for i in range(num_cols)]
                if j % 3
                else [1 for _ in range(num_cols)]
            )
            for j in range(num_rows)
        ]

        # Directed graph for traffic flow + traffic light locations
        self.graph = CityGraph(grid=self.grid).G

        # Place traffic lights at nodes where in-deg == 2
        # Traffic lights start off blocking/stopping/redding avenues, switch every 2 ticks
        self.lights = [node for node, in_deg in self.graph.in_degree() if in_deg == 2]
        self.stop_road = "S"  # either A (avenue) or S (street)
        for node in self.lights:
            self.grid[node[0]][node[1]] = self.stop_road

        # Init time
        # Start off at -1 because we call env_step in the game loop first before showing the city
        self.time = -1

        assert (
            self.grid[spawn_point[0]][spawn_point[1]] == 1
        ), "Player must spawn on a free road, not in a building or at a traffic light."

        # Set player start coordinates / spawn point
        self.p_row, self.p_col = spawn_point
        self.grid[self.p_row][self.p_col] = "P"
        self.prev_cell = 1

        # Init task and task done state
        # Start off at task_idx = 0 (1-indexed) because we call env_step in the game loop first before showing the city
        self.task = (
            ""  # description of task exposed to policy, e.g. "turn left in 2 blocks"
        )
        self._task_directions = []  # golden list of actions (WASD)
        self._task_idx = 0
        self._dir_idx = 0
        self._need_new_task = True

        # Keep track of number of consecutive no-ops to punish no-oping for too long
        self._noops = 0

    def env_step(self):
        """
        Handles natural environment changes not caused by player actions.
        """
        # Move time forward by 1 tick
        self.time += 1

        # Change traffic light every 2 ticks
        if self.time % 2 == 0:
            self.stop_road = "A" if self.stop_road == "S" else "S"
            for node in self.lights:
                if self.grid[node[0]][node[1]] == "P":
                    # player is at that node, don't replace it on the grid
                    continue
                self.grid[node[0]][node[1]] = self.stop_road

        if self._need_new_task:
            self.task, self._task_directions = self._generate_task()
            self._task_idx += 1
            self._dir_idx = 0
            self._need_new_task = False

    def step(self, action):
        """
        Handles consequences of player actions.

        Args:
        - action: str -- "W" (move North), "A" (move West), "S" (move South), "D" (move East)
        """
        action = action.strip().upper()
        action = "" if action == "<BLANK>" else action.removesuffix("_ACT")

        assert action in [
            "W",
            "A",
            "S",
            "D",
            "",
        ], "Player can only move North/South/East/West or no-op."

        future_row, future_col = self.p_row, self.p_col
        match action:
            case "W":
                future_row -= 1
            case "A":
                future_col -= 1
            case "S":
                future_row += 1
            case "D":
                future_col += 1

        rew = 0
        termination = False

        if not self.graph.has_node((future_row, future_col)):
            # player attempts to exit road, e.g. go off grid, go into building
            # severe punishment, terminate episode
            rew -= self.SEVERE_REW
            termination = True
            return rew, termination  # return early for illegal movements

        if (self.p_row, self.p_col) == (future_row, future_col):
            # player no-ops (stops)
            self._noops += 1

            if self._noops == 5:
                # stays stationary for too long
                # medium punishment (blocking traffic), continue episode
                rew -= self.MEDIUM_REW

            if self.prev_cell != 1:
                # stops at a traffic light intersection
                # medium punishment (blocking traffic), continue episode
                rew -= self.MEDIUM_REW
            elif (
                self.graph.has_edge(
                    (self.p_row, self.p_col), (self.p_row - 1, self.p_col)
                )
                and self.grid[self.p_row - 1][self.p_col] == "A"
            ):
                # in front of an active red light (avenue north-bound)
                # normal reward, continue episode
                rew += self.NORMAL_REW
            elif (
                self.graph.has_edge(
                    (self.p_row, self.p_col), (self.p_row + 1, self.p_col)
                )
                and self.grid[self.p_row + 1][self.p_col] == "A"
            ):
                # in front of an active red light (avenue south-bound)
                # normal reward, continue episode
                rew += self.NORMAL_REW
            elif (
                self.graph.has_edge(
                    (self.p_row, self.p_col), (self.p_row, self.p_col + 1)
                )
                and self.grid[self.p_row][self.p_col + 1] == "S"
            ):
                # in front of an active red light (street east-bound)
                # normal reward, continue episode
                rew += self.NORMAL_REW
            elif (
                self.graph.has_edge(
                    (self.p_row, self.p_col), (self.p_row, self.p_col - 1)
                )
                and self.grid[self.p_row][self.p_col - 1] == "S"
            ):
                # in front of an active red light (street west-bound)
                # normal reward, continue episode
                rew += self.NORMAL_REW

            return rew, termination  # return early for no-op

        self._noops = 0  # reset consecutive no-op count

        # Parallel IF blocks to handle multiple traffic rules being broken simultaneously when player is moving
        # Example: driving in the opposite lane while running an active red light
        # Above early returns allow these IF blocks to be written in parallel without concern

        if self.grid[future_row][future_col] == "A" and (
            action == "W" or action == "S"
        ):
            # player runs traffic light (avenue)
            # medium punishment (law broken), continue episode
            rew -= self.MEDIUM_REW
        elif self.grid[future_row][future_col] == "S" and (
            action == "A" or action == "D"
        ):
            # player runs traffic light (street)
            # medium punishment (law broken), continue episode
            rew -= self.MEDIUM_REW

        if self.graph.has_edge((self.p_row, self.p_col), (future_row, future_col)):
            # player follows traffic flow (not driving in the opposite lane)
            # normal reward, continue episode
            rew += self.NORMAL_REW
            self._move_player(future_row, future_col)
        else:
            # player does not follow traffic flow (driving in the opposite lane, U-turning)
            # medium punishment (law broken), continue episode
            rew -= self.MEDIUM_REW
            self._move_player(future_row, future_col)

        # Handle instruction-following (or lack thereof)
        if action:  # not no-op, since we don't want to punish waiting for red light
            if action != self._task_directions[self._dir_idx]:
                # deviate from instruction
                # medium punishment, get new instructions, continue episode
                rew -= self.MEDIUM_REW
                self._need_new_task = True
            elif self._dir_idx == len(self._task_directions) - 1:
                # fulfilled current task in its entirety
                # medium reward, get new instructions, continue episode
                rew += self.MEDIUM_REW
                self._need_new_task = True
                if self._task_idx == self.MAX_TASKS:
                    # terminate once MAX_TASKS tasks are fulfilled
                    termination = True
            self._dir_idx += 1

        return rew, termination

    def _move_player(self, future_row, future_col):
        self.grid[self.p_row][self.p_col] = (
            self.prev_cell if self.prev_cell == 1 else self.stop_road
        )
        self.prev_cell = self.grid[future_row][future_col]
        self.p_row, self.p_col = future_row, future_col
        self.grid[self.p_row][self.p_col] = "P"

    def _generate_task(self):
        # pick one random out-edge from the player node
        curr_row, curr_col = self.p_row, self.p_col
        out_edge = random.choice(list(self.graph.out_edges((curr_row, curr_col))))

        # get direction (WASD)
        diff_row, diff_col = out_edge[1][0] - self.p_row, out_edge[1][1] - self.p_col
        direction = ""
        num_repeat = 0
        if diff_row == -1:
            direction = "W"
        elif diff_row == 1:
            direction = "S"
        elif diff_col == -1:
            direction = "A"
        elif diff_col == 1:
            direction = "D"

        # follow direction until the current node no longer has an out-edge that is in that direction
        # keeping track of nodes with out-deg > 1 and final terminal node of the avenue/street (water-adjacent)

        # key is intersection coord, val is number of times current direction is repeated to reach the intersection
        intersections = {}
        while self.graph.has_edge(
            (curr_row, curr_col), (curr_row + diff_row, curr_col + diff_col)
        ):
            num_repeat += 1
            curr_row += diff_row
            curr_col += diff_col
            if len(self.graph.out_edges((curr_row, curr_col))) > 1:
                intersections[(curr_row, curr_col)] = num_repeat
        intersections[(curr_row, curr_col)] = num_repeat

        # pick random intersection, randomly pick different out-edge (new direction / turn)
        intersection = random.choice(list(intersections.keys()))
        new_out_edges = list(self.graph.out_edges((intersection[0], intersection[1])))
        new_out_edge = None
        if len(new_out_edges) == 1:
            new_out_edge = new_out_edges[0]
        else:
            new_out_edges.remove(
                (
                    (intersection[0], intersection[1]),
                    (intersection[0] + diff_row, intersection[1] + diff_col),
                )
            )
            new_out_edge = random.choice(new_out_edges)

        # get new direction (turn)
        target = new_out_edge[1]
        new_diff_row, new_diff_col = (
            target[0] - intersection[0],
            target[1] - intersection[1],
        )
        turn = ""
        if new_diff_row == -1:
            turn = "W"
        elif new_diff_row == 1:
            turn = "S"
        elif new_diff_col == -1:
            turn = "A"
        elif new_diff_col == 1:
            turn = "D"

        directions = [direction] * intersections[intersection] + [turn]

        # build task description
        ave_st = "Avenue" if turn == "W" or turn == "S" else "Street"

        road_number = 0
        if ave_st == "Avenue":
            if target[1] == 0:
                road_number = 4
            elif target[1] == 3 or target[1] == 4:
                road_number = 3
            elif target[1] == 7 or target[1] == 8:
                road_number = 2
            else:
                road_number = 1
        else:
            road_number = 6 - target[0] // 3
        ordinal = ""
        if road_number == 1:
            ordinal = "st"
        elif road_number == 2:
            ordinal = "nd"
        elif road_number == 3:
            ordinal = "rd"
        else:
            ordinal = "th"

        left_right = None  # first-person perspective
        if (
            (direction == "W" and turn == "D")
            or (direction == "S" and turn == "A")
            or (direction == "D" and turn == "S")
            or (direction == "A" and turn == "W")
        ):
            left_right = "right"
        else:
            left_right = "left"

        units = 0
        if turn == "W" or turn == "S":
            units = abs(target[1] - self.p_col)
        else:
            units = abs(target[0] - self.p_row)

        description = (
            f"Turn {left_right} in {units} unit{'s' if units > 1 else ''}."
            if random.random() > 0.5
            else f"Turn {left_right} on {road_number}{ordinal} {ave_st}."
        )

        return description, directions

    def _color(self, cell):
        if cell == 0:
            return colored(cell, "blue")
        elif cell == 1:
            return str(cell)
        elif cell == "P":
            return colored(cell, "yellow")
        elif cell == "A" or cell == "S":
            return colored(cell, "red")

    def __str__(self):
        out = "\n".join([" ".join([self._color(x) for x in hoz]) for hoz in self.grid])
        return out


if __name__ == "__main__":
    while True:
        try:
            city = City(spawn_point=(random.randrange(16), random.randrange(12)))
            break
        except AssertionError:
            continue

    termination = False

    while not termination:
        city.env_step()  # placing env_step here necessitates init-ing time at -1

        print(f"[{city._task_idx}/{city.MAX_TASKS}] {city.task}")
        print(city._noops)
        print(city)
        action = input("::").upper()
        rew, termination = city.step(action)

        print(f"Reward: {rew}")
        print("\n\n")
