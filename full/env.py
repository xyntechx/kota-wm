from graph import CityGraph
from termcolor import colored


class City:
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
        for node in self.lights:
            self.grid[node[0]][node[1]] = "A"

        assert (
            self.grid[spawn_point[0]][spawn_point[1]] == 1
        ), "Player must spawn on a free road, not in a building or at a traffic light."

        # Set player start coordinates / spawn point
        self.p_row, self.p_col = spawn_point
        self.grid[self.p_row][self.p_col] = "P"
        self.prev_cell = 1

    def step(self, action):
        """
        Args:
        - action: str -- "W" (move North), "A" (move West), "S" (move South), "D" (move East)
        """
        # TODO: reward (incl task fulfillment), traffic light switch

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
            rew -= 10
            termination = True
        elif self.graph.has_edge((self.p_row, self.p_col), (future_row, future_col)):
            # player follows traffic flow (not driving in the opposite lane)
            # normal reward, continue episode
            rew += 1
            self._move_player(future_row, future_col)
        else:
            # player does not follow traffic flow (driving in the opposite lane)
            # normal punishment, continue episode
            rew -= 1
            self._move_player(future_row, future_col)

        return rew, termination

    def _move_player(self, future_row, future_col):
        self.grid[self.p_row][self.p_col] = self.prev_cell
        self.prev_cell = self.grid[future_row][future_col]
        self.p_row, self.p_col = future_row, future_col
        self.grid[self.p_row][self.p_col] = "P"

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
    city = City(spawn_point=(4, 8))
    termination = False

    while not termination:
        print(city)
        action = input("::")
        rew, termination = city.step(action)
        print(f"Reward: {rew}")
