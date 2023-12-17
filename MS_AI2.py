# Minesweeper AIs

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Minesweeper import *
from functools import reduce
from itertools import combinations

############################################################
# Helper Functions
############################################################

def get_connected_subgraphs(G, size):
    # https://stackoverflow.com/a/75873085/14165696
    """Get all connected subgraphs by a recursive procedure"""

    con_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True) if len(c) >= size]

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

def generate_mine_perms(n, m):
    def rightmost(ls, x):
        for i in range(len(ls)):
            if ls[-(i+1)] == x:
                return len(ls) - (i+1)
        return None

    def slam_right(ls, x):
        count = len([i for i in ls if i == x])
        return [i for i in ls if i != x]  + count*[x]

    def swap(ls, a, b):
        ls = list(ls)
        tmp = ls[a]
        ls[a] = ls[b]
        ls[b] = tmp
        return ls
    
    current = (n-m)*[0] + m*[1]
    final = m*[1] + (n-m)*[0]
    yield current
    while not np.all(np.array(current) == np.array(final)):
        # Find the rightmost 0 to the left of the rightmost 1.
        rightmost0 = rightmost(current[:rightmost(current, 1)], 0)

        # move a 1 from the right into that 0
        current = swap(current, rightmost0, current[rightmost0:].index(1) + rightmost0)
        yield current

        # move all the other 1s to the right of that zero as far right as possible.
        slammed = current[:rightmost0+1] + slam_right(current[rightmost0+1:], 1)
        if not np.all(np.array(current) == np.array(slammed)):
            current = slammed
            yield current

############################################################
# The Best AI Ever
############################################################

class MS_AI:
    def __init__(self, game : MS_Game2):
        self.game = game

        game.TileDisplayHandlers += [self.handle_tile_display]
        game.BulkTileDisplayHandlers += [self.handle_bulk_tile_display]
        game.TileFlagChangedHandlers += [self.handle_tile_flag_changed]
        game.GameCompleteHandlers += [self.handle_game_complete]
        game.GameResetHandlers += [self.handle_game_reset]

        self.full_graph = nx.Graph()
        self.number_graph = nx.Graph()

    def handle_tile_display(self, tile):
        # update number graph after full graph
        self.update_full_graph(tile)
        self.update_number_graph(tile)

    def handle_bulk_tile_display(self, tiles):
        for tile in tiles:
            self.handle_tile_display(tile)

    def handle_tile_flag_changed(self, id):
        self.flag_update_full_graph(id)
        self.flag_update_number_graph(id)

    def handle_game_complete(self):
        foo = 1

    def handle_game_reset(self):
        foo = 1

    def update_full_graph(self, tile):
        # if it's in the full graph, it must be from when it was un-displayed.
        # Remove to also remove edges to other displayed tiles.
        if tile.id in self.full_graph.nodes:
            self.full_graph.remove_node(tile.id)

        if tile.mine_count == 0:
            return
        
        neighbor_tiles = self.game.get_tiles(tile.neighbors)
        flagged_neighbors = len([n for n in neighbor_tiles if n.flagged])
        tile.effective_count = tile.mine_count - flagged_neighbors

        if tile.effective_count <= 0:
            return

        self.full_graph.add_node(tile.id, **vars(tile))
        for neighbor in self.game.get_tiles(tile.neighbors):
            if not neighbor.displayed:
                self.full_graph.add_node(neighbor.id, **vars(neighbor))
                self.full_graph.add_edge(tile.id, neighbor.id)

    def update_number_graph(self, tile):
        if tile.mine_count == 0 or tile.effective_count <= 0:
            return
        
        neighbor_tiles = self.game.get_tiles(tile.neighbors)

        # check for edges that need to be modified
        for a,b,attr in list(self.number_graph.edges.data()):
            if not tile.id in attr["nodes"]:
                continue
            attr["nodes"].remove(tile.id)
            if len(attr["nodes"]) == 0:
                self.number_graph.remove_edge(a,b)
        
        # add new number to number graph
        self.number_graph.add_node(tile.id, **vars(tile))

        # add edges
        for uncleared_neighbor in self.full_graph[tile.id]:
            for number_neighbor in self.full_graph[uncleared_neighbor]:
                if number_neighbor == tile.id:
                    continue
                if self.number_graph.has_edge(tile.id, number_neighbor):
                    self.number_graph[tile.id][number_neighbor]["nodes"].append(uncleared_neighbor)
                else:
                    self.number_graph.add_node(number_neighbor, **self.full_graph.nodes[number_neighbor])
                    self.number_graph.add_edge(tile.id, number_neighbor, nodes=[uncleared_neighbor])

    def flag_update_full_graph(self, id):
        if not id in self.full_graph.nodes:
            return
        self.full_graph.nodes[id]["flagged"] = self.game.is_flagged(id)
        for neb in self.full_graph[id]:
            self.full_graph.nodes[neb]["effective_count"] += -1 if self.game.is_flagged(id) else 1

    def flag_update_number_graph(self, id):
        flagged = self.game.is_flagged(id)
        for number_tile in self.full_graph[id]:
            self.number_graph.nodes[number_tile]["effective_count"] += -1 if flagged else 1
            if self.number_graph.nodes[number_tile]["effective_count"] == 0:
                self.number_graph.remove_node(number_tile)

    def reset(self):
        self.full_graph = nx.Graph()

    def display(self):
        number_nodes = [id for id,attr in self.full_graph.nodes(data=True) if attr["displayed"]]
        flagged_nodes = [id for id,attr in self.full_graph.nodes(data=True) if not attr["displayed"] and attr["flagged"]]
        unflagged_nodes = [id for id,attr in self.full_graph.nodes(data=True) if not attr["displayed"] and not attr["flagged"]]
        for n in number_nodes:
            self.full_graph.nodes[n]["color"] = 'blue'
        for f in flagged_nodes:
            self.full_graph.nodes[f]["color"] = 'yellow'
        for u in unflagged_nodes:
            self.full_graph.nodes[u]["color"] = 'red'

        components = [self.full_graph.subgraph(c).copy() for c in nx.connected_components(self.full_graph)]
        n = len(components)
        fig, ax = plt.subplots(ncols=n+1)
        ax = ax if hasattr(ax, '__iter__') else [ax]
        for i in range(n):
            plt.figure(i+1)
            graph = components[i]
            scale = len(graph.nodes) / len(self.full_graph.nodes)
            pos = nx.bipartite_layout(graph, [n for n in number_nodes if n in graph.nodes], align='horizontal', scale=scale)
            for id,coord in pos.items():
                x,y = coord
                if id in number_nodes:
                    node_data = graph.nodes[id]
                    ax[i].text(x,y-0.1,s="{} ({})".format(node_data["mine_count"], node_data["effective_count"]), horizontalalignment='center')
            node_color = [attr["color"] for id,attr in graph.nodes(data=True)]
            nx.draw(graph, pos=pos, with_labels=True, node_color=node_color, ax=ax[i])

        nx.draw(self.number_graph, with_labels=True, ax=ax[-1])
        fig.show()

    def level_zero_actions(self):
        tiles_to_flag = []
        tiles_to_display = []

        number_nodes = [id for id,attr in self.full_graph.nodes(data=True) if attr["displayed"]]
        for id in number_nodes:
            unflagged_neighbors = [id for id in self.full_graph[id] if not self.game.is_flagged(id)]

            if self.full_graph.nodes.data()[id]["effective_count"] == 0:
                tiles_to_display += [neb for neb in unflagged_neighbors if not neb in tiles_to_display]
            elif self.full_graph.nodes.data()[id]["effective_count"] == len(unflagged_neighbors):
                tiles_to_flag += [neb for neb in unflagged_neighbors if not neb in tiles_to_flag]

        return tiles_to_flag, tiles_to_display
    
    def level_n_actions(self, n):
        if n == 0:
            return self.level_zero_actions()
        
        tiles_to_display = []
        tiles_to_flag = []

        for subgraph_indices in get_connected_subgraphs(self.number_graph, n):
            number_subgraph = nx.subgraph(self.number_graph, subgraph_indices)
            
            # determine number of mines to try to apply to this subgraph
            mine_count = min([self.number_graph.nodes[i]["effective_count"] for i in subgraph_indices])
            
            # determine union of all adjacent uncleared tiles
            for i in subgraph_indices:
                for id in self.full_graph[i]:
                    if not self.full_graph.nodes[id]["flagged"]:
                        continue
                    foo = 1
            mineable_tiles = list(reduce(lambda x,y: x.union(y), [set([id for id in self.full_graph[i] if not self.full_graph.nodes[id]["flagged"]]) for i in subgraph_indices]))
            
            # determine all valid allocations of mines to tiles
            valid_perms = []
            tile_count = len(mineable_tiles)
            for perm in generate_mine_perms(len(mineable_tiles), mine_count):
                valid = True
                for num_tile in subgraph_indices:
                    try:
                        unflagged_neighbors = [id for id in self.full_graph[num_tile] if not self.full_graph.nodes[id]["flagged"]]
                        mine_count = len([id for id in unflagged_neighbors if perm[mineable_tiles.index(id)] == 1])
                    except:
                        foo = 1
                    if self.number_graph.nodes[num_tile]["effective_count"] != mine_count:
                        break
                if valid:
                    valid_perms.append(perm)
            
            # determine any tiles that are always mined/not-mined
            always_mined = tile_count*[True]
            always_clear = tile_count*[True]

            for p in valid_perms:
                for i in range(tile_count):
                    if p[i] == 0:
                        always_mined[i] = False
                    else:
                        always_clear[i] = False
            
            # add to flag/clear lists
            for i in range(tile_count):
                if always_mined[i]:
                    tiles_to_flag.append[mineable_tiles[i]]
                if always_clear[i]:
                    tiles_to_display.append(mineable_tiles[i])
        
        return tiles_to_flag, tiles_to_display


