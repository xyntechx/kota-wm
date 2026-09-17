import networkx as nx


class CityGraph:
    def __init__(self, grid):
        num_rows, num_cols = len(grid), len(grid[0])

        # Init nodes (row_idx, col_idx)
        road_cells = []
        for i in range(num_rows):
            for j in range(num_cols):
                if grid[i][j] == 1:  # exclude buildings
                    road_cells.append((i, j))
        self.G = nx.DiGraph()
        self.G.add_nodes_from(road_cells)

        # Avenues
        avenues = []
        for j in range(num_cols):
            if j % 4 == 0:
                # North-bound
                avenues.extend([((i, j), (i - 1, j)) for i in range(1, num_rows)])
            elif (j + 1) % 4 == 0:
                # South-bound
                avenues.extend([((i, j), (i + 1, j)) for i in range(0, num_rows - 1)])
        self.G.add_edges_from(avenues)

        # Streets
        streets = []
        for i in range(num_rows):
            if i % 6 == 0:
                # East-bound
                streets.extend([((i, j), (i, j + 1)) for j in range(0, num_cols - 1)])
            elif (i + 3) % 6 == 0:
                # West-bound
                streets.extend([((i, j), (i, j - 1)) for j in range(1, num_cols)])
        self.G.add_edges_from(streets)
