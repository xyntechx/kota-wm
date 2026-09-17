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
            self.grid[spawn_point[0]][spawn_point[1]] == 0
        ), "Player must spawn in a building, not on a road."

        # Set player start coordinates / spawn point
        self.player_row, self.player_col = spawn_point
        self.grid[self.player_row][self.player_col] = "P"

    def step(self, action):
        # TODO: reward (incl task fulfillment), traffic light switch
        pass


if __name__ == "__main__":

    def draw_color(cell):
        if cell == 0:
            return colored(cell, "blue")
        elif cell == 1:
            return str(cell)
        elif cell == "P":
            return colored(cell, "yellow")
        elif cell == "A" or cell == "S":
            return colored(cell, "red")

    city = City(spawn_point=(2, 9))
    for hoz in city.grid:
        print(" ".join([draw_color(x) for x in hoz]))
