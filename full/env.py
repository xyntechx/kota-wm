from termcolor import colored


class City:
    def __init__(self, spawn_point=(0, 0)):
        """
        City with 6 streets and 4 avenues, where streets cut horizontally and are one-way, while avenues cut vertically and are two-way
        """

        num_rows, num_cols = 16, 12
        building_ys = (1, 2, 5, 6, 9, 10)
        assert (
            0 <= spawn_point[0] < num_rows and 0 <= spawn_point[1] < num_cols
        ), "Player cannot spawn off-grid; grid has 16 rows, 12 cols."

        # Init city -- 0: building, 1: road, 2: player
        self.grid = [
            (
                [0 if i in building_ys else 1 for i in range(num_cols)]
                if j % 3
                else [1 for _ in range(num_cols)]
            )
            for j in range(num_rows)
        ]

        assert (
            self.grid[spawn_point[0]][spawn_point[1]] == 1
        ), "Player must spawn on a road, not in a building."

        # Init player start coordinates / spawn point
        self.player_row, self.player_col = spawn_point
        self.grid[self.player_row][self.player_col] = 2

    def step(self, action):
        pass


if __name__ == "__main__":
    city = City(spawn_point=(14, 10))
    for hoz in city.grid:
        print(" ".join([colored(x, "blue") if not x else str(x) for x in hoz]))
