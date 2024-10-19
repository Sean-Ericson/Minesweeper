# Minesweeper AIs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from Minesweeper import *
from functools import reduce
from itertools import combinations
from networkx.algorithms import bipartite

############################################################
# Helper Functions
############################################################

def get_connected_subgraphs(G, size):
    # https://stackoverflow.com/a/75873085/14165696
    """Get all connected subgraphs by a recursive procedure"""
    def recursive_local_expand(node_set, possible, excluded, results, max_size):
        """
        Recursive function to add an extra node to the subgraph being formed
        """
        if len(node_set) == max_size:
            results.append(list(node_set))
            return
        for j in possible - excluded:
            new_node_set = node_set | {j}
            excluded = excluded | {j}
            new_possible = (possible | set(G.neighbors(j))) - excluded
            recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)
   
    results = []
    excluded = set()
    for i in G:
        excluded.add(i)
        recursive_local_expand({i}, set(G.neighbors(i)) - excluded, excluded, results, size)

    return results

############################################################
# Helper Classes
############################################################

class Number_Subgraph:
    def __init__(self, nodes, level=1):
        self.nodes = nodes
        self.level = level

class FullGraph(nx.Graph):
    def __init__(self, field: MS_Field) -> None:
        super().__init__()
        self.field = field

    def tile_displayed(self, tile: MS_Tile):
        # if it's in the full graph, it must be from before it was displayed.
        # Remove to also remove edges to other displayed tiles.
        if tile.id in self.nodes:
            self.remove_node(tile.id)

        if tile.mine_count == 0:
            return

        self.add_node(tile.id, tile=tile)
        for neighbor in self.field.neighbors(tile.id):
            if not neighbor.displayed:
                self.add_node(neighbor.id, tile=neighbor)
                self.add_edge(tile.id, neighbor.id)

############################################################
# The Best AI Ever
############################################################

class MS_AI:
    def __init__(self, game : MS_Game) -> None:
        game.e_TilesDisplayed += self._MS_Game_TilesDisplayed
        game.e_TileFlagChanged += self._MS_Game_TileFlagChanged
        game.e_GameReset += self._MS_Game_GameReset
        self.game = game

        self.full_graph = FullGraph(game.field)

    def _level_one_actions(self):
        tiles_to_flag = []
        tiles_to_display = []

        number_nodes = [id for id,attr in self.full_graph.nodes(data=True) if attr["tile"].displayed]
        for id in number_nodes:
            unflagged_neighbors = [id for id in self.full_graph[id] if not self.game.field[id].flagged]

            if self.full_graph.nodes.data()[id]["tile"].effective_count == 0:
                tiles_to_display += [neb for neb in unflagged_neighbors if not neb in tiles_to_display]
            elif self.full_graph.nodes.data()[id]["tile"].effective_count == len(unflagged_neighbors):
                tiles_to_flag += [neb for neb in unflagged_neighbors if not neb in tiles_to_flag]

        return tiles_to_flag, tiles_to_display

    def _make_number_graph(self):
        number_tiles = []
        for i in self.full_graph.nodes:
            tile = self.full_graph.nodes[i]["tile"]
            if tile.displayed and tile.effective_count > 0:
                number_tiles.append(i)
        return bipartite.projected_graph(self.full_graph, number_tiles)

    def _MS_Game_TilesDisplayed(self, tiles: list[MS_Tile]) -> None:
        for tile in tiles:
            self.full_graph.tile_displayed(tile)

    def _MS_Game_TileFlagChanged(self, tile: MS_Tile):
        pass

    def _MS_Game_GameReset(self) -> None:
        self.reset()

    def display_full_graph(self) -> None:
        full_graph = nx.Graph(self.full_graph)
        number_nodes = [id for id,attr in full_graph.nodes(data=True) if attr["tile"].displayed]
        flagged_nodes = [id for id,attr in full_graph.nodes(data=True) if not attr["tile"].displayed and attr["tile"].flagged]
        unflagged_nodes = [id for id,attr in full_graph.nodes(data=True) if not attr["tile"].displayed and not attr["tile"].flagged]
        for n in number_nodes:
            full_graph.nodes[n]["color"] = 'blue'
        for f in flagged_nodes:
            full_graph.nodes[f]["color"] = 'yellow'
        for u in unflagged_nodes:
            full_graph.nodes[u]["color"] = 'red'

        components0 = [full_graph.subgraph(c).copy() for c in nx.connected_components(full_graph)]
        components = []
        for c in components0:
            dead = True
            for i, attr in c.nodes.data():
                if not attr["tile"].displayed:
                    continue
                if attr["tile"].effective_count > 0:
                    dead = False
                    break
            if not dead:
                components.append(c)

        n = len(components)
        fig, ax = plt.subplots(ncols=n)
        ax = ax if hasattr(ax, '__iter__') else [ax]
        for i in range(n):
            plt.figure(i+1)
            graph = components[i]
            scale = len(graph.nodes) / len(full_graph.nodes)
            pos = nx.bipartite_layout(graph, [n for n in number_nodes if n in graph.nodes], align='horizontal', scale=int(scale))
            for id,coord in pos.items():
                x,y = coord
                if id in number_nodes:
                    node_data = graph.nodes[id]
                    ax[i].text(x,y-0.1,s="{} ({})".format(node_data["tile"].mine_count, node_data["tile"].effective_count), horizontalalignment='center')
            node_color = [attr["color"] for id,attr in graph.nodes(data=True)]
            nx.draw(graph, pos=pos, with_labels=True, node_color=node_color, ax=ax[i])
        fig.show()

    def display_number_graph(self) -> None:
        number_graph = self._make_number_graph()
        pos = {}
        sep_x = 2.0 / self.game.x
        sep_y = 2.0 / self.game.y
        for id in number_graph.nodes:
            x, y = self.game.field.tile_loc(id)
            pos[id] = np.array([(x + 0.7*(np.random.random() - 0.5))*sep_x, 1 - (y + 0.7*(np.random.random() - 0.5))*sep_y])

        fig, ax = plt.subplots()
        nx.draw(number_graph, pos=pos, with_labels=True, ax=ax)
        fig.show()
    
    def get_valid_mine_perms(self, number_tiles, mineable_tiles, current_mines=None):
        current_mines = current_mines or []
        if isinstance(number_tiles, list):
            number_tiles = {i: self.full_graph.nodes[i]["tile"].effective_count for i in number_tiles}
        if len(mineable_tiles) == 0 or len(number_tiles) == 0:
            yield current_mines
        else:
            num_tile = list(number_tiles.keys())[0]
            neighbor_tiles = [t for t in self.full_graph[num_tile] if t in mineable_tiles]
            for mines in combinations(neighbor_tiles, number_tiles[num_tile]):
                tmp = {i: number_tiles[i] - len([j for j in mines if j in self.full_graph[i]]) for i in number_tiles if not i == num_tile}
                if any([v < 0 for v in tmp.values()]):
                    continue
                new_number_tiles = {k:v for k,v in tmp.items() if v > 0}
                new_mineable_tiles = [x for x in mineable_tiles if not x in neighbor_tiles]
                new_mines = current_mines + list(mines)
                yield from self.get_valid_mine_perms(new_number_tiles, new_mineable_tiles, new_mines)

    def level_n_actions(self, n, nodes=None):
        if n == 1:
            return self._level_one_actions()

        number_graph = self._make_number_graph()
        tiles_to_display = []
        tiles_to_flag = []

        for subgraph_indices in get_connected_subgraphs(number_graph if nodes is None else number_graph.subgraph(nodes), n):

            # determine union of all adjacent uncleared tiles
            mineable_tiles = list(reduce(lambda x,y: x.union(y), [set([id for id in self.full_graph[i] if not self.full_graph.nodes[id]["tile"].flagged]) for i in subgraph_indices]))

            valid_perms = [perm for perm in self.get_valid_mine_perms(subgraph_indices, mineable_tiles)]
            if len(valid_perms) == 0:
                continue

            # add to flag/clear lists
            for mineable_tile in mineable_tiles:
                if all([mineable_tile in perm for perm in valid_perms]) and not mineable_tile in tiles_to_flag:
                    tiles_to_flag.append(mineable_tile)
                if all([not mineable_tile in perm for perm in valid_perms]) and not mineable_tile in tiles_to_display:
                    tiles_to_display.append(mineable_tile)
        
        return tiles_to_flag, tiles_to_display    

    def reset(self) -> None:
        self.full_graph = FullGraph(self.game.field)