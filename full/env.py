import numpy as np
from termcolor import colored


class City:
    def __init__(self):
        """
        City with 6 streets and 4 avenues, where streets cut horizontally and are one-way, while avenues cut vertically and are two-way
        """

        # Init city
        building_ys = (1, 2, 5, 6, 9, 10)
        num_rows, num_cols = 16, 12
        self.grid = np.array(
            [
                (
                    [0 if i in building_ys else 1 for i in range(num_cols)]
                    if j % 3
                    else [1 for _ in range(num_cols)]
                )
                for j in range(num_rows)
            ]
        )
        # streets = np.array(
        #     [
        #         (
        #             [4 if j % 6 else 2 for _ in range(num_cols)]
        #             if not j % 3
        #             else [1 for _ in range(num_cols)]
        #         )
        #         for j in range(0, num_rows)
        #     ]
        # )
        # # self.grid = np.multiply(self.grid, streets)
        # self.grid = np.transpose(np.multiply(self.grid, streets))
        # avenues = np.array(
        #     [
        #         (
        #             [3 if j % 4 else 1 for _ in range(num_rows)]
        #             if not j in building_ys
        #             else [1 for _ in range(num_rows)]
        #         )
        #         for j in range(num_cols)
        #     ]
        # )
        # self.grid = np.transpose(np.multiply(self.grid, avenues))

        # Init player start coordinates -- player spawns at a random spot every game.
        self.player_row = 0
        self.player_col = 0

    def move():
        pass


if __name__ == "__main__":
    city = City()
    for hoz in city.grid:
        print(" ".join([colored(x, "blue") if not x else str(x) for x in hoz]))
