# Minesweeper AIs

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
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

def sample_mine_perms(n, m, num_samples):
    samples = []
    for _ in range(int(num_samples)):
        sample = (n-m)*[0] + m*[1]
        np.random.shuffle(sample)
        samples.append(sample)
    return samples

############################################################
# Helper Classes
############################################################

class Number_Subgraph:
    def __init__(self, nodes, level=1):
        self.nodes = nodes
        self.level = level


############################################################
# The Best AI Ever
############################################################

class MS_AI:
    def __init__(self, game : MS_Game2):
        game.TileDisplayHandlers += [self.handle_tile_display]
        game.TileFlagChangedHandlers += [self.handle_tile_flag_changed]
        game.GameCompleteHandlers += [self.handle_game_complete]
        game.GameResetHandlers += [self.handle_game_reset]
        self.game = game

        self.full_graph = nx.Graph()
        self.number_graph = nx.Graph()
        self.number_subgraphs = []

    def handle_tile_display(self, tiles):
        if not self.game.is_active():
            return
        # update number graph after full graph
        for tile in tiles:
            if self.game.is_mined(tile.id):
                return
            self.update_full_graph(tile)
            self.update_number_graph(tile)
        self.update_number_subgraphs()

    def handle_tile_flag_changed(self, id):
        self.flag_update_full_graph(id)
        self.flag_update_number_graph(id)
        self.update_number_subgraphs()

    def handle_game_complete(self):
        foo = 1

    def handle_game_reset(self):
        self.reset()

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

        #if tile.effective_count <= 0:
        #    return

        self.full_graph.add_node(tile.id, **vars(tile))
        for neighbor in self.game.get_tiles(tile.neighbors):
            if not neighbor.displayed:
                self.full_graph.add_node(neighbor.id, **vars(neighbor))
                self.full_graph.add_edge(tile.id, neighbor.id)

    def update_number_graph(self, tile):
        if tile.mine_count == 0 or tile.effective_count == 0:
            return

        # check for edges that need to be modified
        self.number_graph_remove_undisplayed_tile(tile.id)
        
        # add new number to number graph
        self.number_graph.add_node(tile.id, **vars(tile))

        # add edges
        for uncleared_neighbor in self.full_graph[tile.id]:
            if self.full_graph.nodes[uncleared_neighbor]["flagged"]:
                continue
            for number_neighbor in self.full_graph[uncleared_neighbor]:
                if number_neighbor == tile.id or self.full_graph.nodes[number_neighbor]["effective_count"] == 0:
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
        if not id in self.full_graph.nodes:
            return
        flagged = self.full_graph.nodes[id]["flagged"]
        # change effective counts
        for number_tile in self.full_graph[id]:
            if not number_tile in self.number_graph.nodes:
                continue
            self.number_graph.nodes[number_tile]["effective_count"] += -1 if flagged else 1
            if self.number_graph.nodes[number_tile]["effective_count"] == 0:
                self.number_graph.remove_node(number_tile)
        # check for edges that need to be modified
        if flagged:
            self.number_graph_remove_undisplayed_tile(id)
        else:
            self.number_graph_add_undisplayed_tile(id)
    
    def number_graph_remove_undisplayed_tile(self, undisplayed_tile_id):
        id = undisplayed_tile_id
        for a,b,attr in list(self.number_graph.edges.data()):
                if not id in attr["nodes"]:
                    continue
                attr["nodes"].remove(id)
                if len(attr["nodes"]) == 0:
                    self.number_graph.remove_edge(a,b)

    def number_graph_add_undisplayed_tile(self, undisplayed_tile_id):
        id = undisplayed_tile_id
        number_neighbors = [i for i in self.full_graph[id] if self.full_graph.nodes[i]["effective_count"] > 0]
        for n1, n2 in combinations(number_neighbors, 2):
            if self.number_graph.has_edge(n1, n2):
                if not id in self.number_graph[n1][n2]["nodes"]:
                    self.number_graph[n1][n2]["nodes"].append(id)
            else:
                self.number_graph.add_edge(n1, n2, nodes=[id])

    def update_number_subgraphs(self):
        new_subgraphs = []
        current_subgraph_nodes = list(nx.connected_components(self.number_graph))
        for nodes in current_subgraph_nodes:
            new_graph = True
            for old_subgraph in self.number_subgraphs:
                if old_subgraph.nodes == nodes:
                    new_subgraphs.append(old_subgraph)
                    new_graph = False
                    break
            if new_graph:
                new_subgraphs.append(Number_Subgraph(nodes))
        self.number_subgraphs = new_subgraphs

    def reset(self):
        self.full_graph = nx.Graph()
        self.number_graph = nx.Graph()

    def display_full_graph(self):
        number_nodes = [id for id,attr in self.full_graph.nodes(data=True) if attr["displayed"]]
        flagged_nodes = [id for id,attr in self.full_graph.nodes(data=True) if not attr["displayed"] and attr["flagged"]]
        unflagged_nodes = [id for id,attr in self.full_graph.nodes(data=True) if not attr["displayed"] and not attr["flagged"]]
        for n in number_nodes:
            self.full_graph.nodes[n]["color"] = 'blue'
        for f in flagged_nodes:
            self.full_graph.nodes[f]["color"] = 'yellow'
        for u in unflagged_nodes:
            self.full_graph.nodes[u]["color"] = 'red'

        components0 = [self.full_graph.subgraph(c).copy() for c in nx.connected_components(self.full_graph)]
        components = []
        for c in components0:
            dead = True
            for i, attr in c.nodes.data():
                if not attr["displayed"]:
                    continue
                if attr["effective_count"] > 0:
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
            scale = len(graph.nodes) / len(self.full_graph.nodes)
            pos = nx.bipartite_layout(graph, [n for n in number_nodes if n in graph.nodes], align='horizontal', scale=scale)
            for id,coord in pos.items():
                x,y = coord
                if id in number_nodes:
                    node_data = graph.nodes[id]
                    ax[i].text(x,y-0.1,s="{} ({})".format(node_data["mine_count"], node_data["effective_count"]), horizontalalignment='center')
            node_color = [attr["color"] for id,attr in graph.nodes(data=True)]
            nx.draw(graph, pos=pos, with_labels=True, node_color=node_color, ax=ax[i])
        fig.show()

    def display_number_graph(self):
        pos = {}
        sep_x = 2.0 / self.game.x
        sep_y = 2.0 / self.game.y
        for id in self.number_graph.nodes:
            x, y = self.game.field.tile_loc(id)
            pos[id] = np.array([(x + 0.7*(np.random.random() - 0.5))*sep_x, 1 - (y + 0.7*(np.random.random() - 0.5))*sep_y])

        fig, ax = plt.subplots()
        nx.draw(self.number_graph, pos=pos, with_labels=True, ax=ax)
        fig.show()

    def level_one_actions(self):
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
    
    def level_n_actions(self, n, nodes=None, perm_cutoff=None):
        if n == 1:
            return self.level_one_actions()

        tiles_to_display = []
        tiles_to_flag = []

        for subgraph_indices in get_connected_subgraphs(self.number_graph if nodes is None else self.number_graph.subgraph(nodes), n):

            # determine union of all adjacent uncleared tiles
            mineable_tiles = list(reduce(lambda x,y: x.union(y), [set([id for id in self.full_graph[i] if not self.full_graph.nodes[id]["flagged"]]) for i in subgraph_indices]))
            
            # determine number of mines to try to apply to this subgraph
            #min_mine_count, max_mine_count = self.determine_min_max_mines(subgraph_indices, mineable_tiles)

            valid_perms = [perm for perm in self.get_valid_mine_perms3(subgraph_indices, mineable_tiles)]
            if len(valid_perms) == 0:
                continue

            # add to flag/clear lists
            for mineable_tile in mineable_tiles:
                if all([mineable_tile in perm for perm in valid_perms]) and not mineable_tile in tiles_to_flag:
                    tiles_to_flag.append(mineable_tile)
                if all([not mineable_tile in perm for perm in valid_perms]) and not mineable_tile in tiles_to_display:
                    tiles_to_display.append(mineable_tile)
        
        return tiles_to_flag, tiles_to_display    
        
    def get_valid_mine_perms2(self, number_tiles, mineable_tiles, current_mines=None):
        current_mines = current_mines or []
        if len(number_tiles) == 0:
            yield current_mines
        else:
            num_tile = number_tiles[0]
            neighbor_tiles = [t for t in self.full_graph[num_tile] if t in mineable_tiles]
            effective_count = self.number_graph.nodes[num_tile]["effective_count"] - len([x for x in current_mines if x in self.full_graph[num_tile]])
            if effective_count < 0:
                return
            for nebs in combinations(neighbor_tiles, effective_count):
                new_number_tiles = list(set(number_tiles) - set([num_tile]))
                new_mineable_tiles = [x for x in mineable_tiles if not x in neighbor_tiles]
                new_mines = current_mines + list(nebs)
                yield from self.get_valid_mine_perms2(new_number_tiles, new_mineable_tiles, new_mines)

    def get_valid_mine_perms3(self, number_tiles, mineable_tiles, current_mines=None):
        current_mines = current_mines or []
        if isinstance(number_tiles, list):
            number_tiles = {i: self.full_graph.nodes[i]['effective_count'] for i in number_tiles}
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
                yield from self.get_valid_mine_perms3(new_number_tiles, new_mineable_tiles, new_mines)

    def random_valid_mine_perm(self, number_tiles, mineable_tiles):
        mineable_tiles = list(mineable_tiles)
        while True:
            num_tiles = {i: self.full_graph.nodes[i]['effective_count'] for i in number_tiles}
            mines = []
            while len(num_tiles) > 0 and len(mineable_tiles) > 0 and len(mines) <= self.game.flags:
                num_tile = np.random.choice(list(num_tiles.keys()))
                neighbor_tiles = [t for t in self.full_graph[num_tile] if t in mineable_tiles]
                if num_tiles[num_tile] > len(neighbor_tiles):
                    break
                combos = list(combinations(neighbor_tiles, num_tiles[num_tile]))
                m = combos[np.random.randint(len(combos))]
                tmp = {i: num_tiles[i] - len([j for j in m if j in self.full_graph[i]]) for i in num_tiles if not i == num_tile}
                if any([v < 0 for v in tmp.values()]):
                    continue
                num_tiles = {k:v for k,v in tmp.items() if v > 0}
                mines += list(m)
                mineable_tiles = [x for x in mineable_tiles if not x in neighbor_tiles]
            
            effective_counts = [self.full_graph.nodes[i]['effective_count'] - len([j for j in mines if j in self.full_graph[i]]) for i in num_tiles]
            if all([c == 0 for c in effective_counts]):
                return mines

    def sample_valid_mine_perms2(self, number_tiles, mineable_tiles, samples):
        return [self.random_valid_mine_perm(number_tiles, mineable_tiles) for _ in range(samples)]

    def sample_valid_mine_perms(self, number_tiles, mineable_tiles, min_mine_count, max_mine_count, num_samples=1000):
        samples = []
        i = 0
        while len(samples) < num_samples:
            i += 1
            nums = list(number_tiles)
            valid_sample = True
            sample = []
            while len(nums) > 0 and len(sample) < self.game.flags:
                np.random.shuffle(nums)
                num = nums.pop()
                neighbor_tiles = [t for t in self.full_graph[num] if t in mineable_tiles]
                effective_count = self.number_graph.nodes[num]["effective_count"] - len([x for x in sample if x in self.full_graph[num]])
                if effective_count <= 0:
                    valid_sample = False
                    break
                combs = list(combinations(neighbor_tiles, min(effective_count, self.game.flags)))
                nebs = combs[np.random.randint(len(combs))]
                sample += list(nebs)
                sample = list(set(sample))

                for num_tile in number_tiles:
                    neighbor_mine_count = len([id for id in self.full_graph[num_tile] if id in sample])
                    if len(nums) > 0:
                        if self.number_graph.nodes[num_tile]["effective_count"] < neighbor_mine_count:
                            valid_sample = False
                            break
                    else:
                        if self.number_graph.nodes[num_tile]["effective_count"] != neighbor_mine_count:
                            valid_sample = False
                            break
            
            if valid_sample:
                samples.append(sample)
        print("sample prop: {}".format(i/num_samples))
        return samples

                
        samples = []
        while len(samples) < num_samples:
            num_mines = np.random.randint(min_mine_count, max_mine_count+1)
            perm = (len(mineable_tiles) - num_mines)*[False] + num_mines*[True]
            np.random.shuffle(perm)
            valid = True
            for num_tile in number_tiles:
                neighbor_mine_count = len([id for id in self.full_graph[num_tile] if id in mineable_tiles and perm[mineable_tiles.index(id)]])
                if self.number_graph.nodes[num_tile]["effective_count"] != neighbor_mine_count:
                    valid = False
                    break
            if valid:
                far = "var"
                samples.append([id for id in mineable_tiles if perm[mineable_tiles.index(id)]])

        return samples

    def perform_actions(self, flags, clears):
        for id in flags:
            self.game.flag(int(id))
        for id in clears:
            if not self.game.is_active():
                return
            self.game.clear(int(id))

    def do_deductive_action(self, max_level):
        # flag/clear as much as possible
        actionable_subgraphs = sorted([g for g in self.number_subgraphs if g.level <= len(g.nodes) and g.level <= max_level], key=lambda x: x.level)
        while self.game.is_active() and self.game.flags > 0 and len(actionable_subgraphs) > 0:
            g = actionable_subgraphs[0]
            flags, clears = self.level_n_actions(g.level, nodes=g.nodes)
            if len(flags) > 0 or len(clears) > 0:
                self.perform_actions(flags, clears)
                g.level = 1
            else:
                g.level += 1
            actionable_subgraphs = sorted([g for g in self.number_subgraphs if g.level <= len(g.nodes) and g.level <= max_level], key=lambda x: x.level)

    def determine_min_max_mines(self, number_tiles, mineable_tiles):
        """ des_num_tiles = sorted(list(number_tiles), key=lambda x: self.number_graph.nodes[x]["effective_count"], reverse=True)

        number_subset = []
        covered_tiles = []
        while not set(covered_tiles) == set(mineable_tiles):
            next_num_tile = des_num_tiles.pop()
            if set(self.full_graph[next_num_tile]).issubset(set(covered_tiles)):
                continue
            number_subset.append(next_num_tile)
            covered_tiles += self.full_graph[next_num_tile]
        
        min_mine_count = min(max([self.number_graph.nodes[i]["effective_count"] for i in number_tiles]), self.game.flags)
        max_mine_count = min(sum([self.number_graph.nodes[i]["effective_count"] for i in number_subset]), self.game.flags) """

        min_mine_count = min(max([self.number_graph.nodes[i]["effective_count"] for i in number_tiles]), self.game.flags)
        max_mine_count = min(sum([self.number_graph.nodes[i]["effective_count"] for i in number_tiles]), len(mineable_tiles), self.game.flags)

        return min_mine_count, max_mine_count

    def do_probable_actions(self, samples, yolo_cutoff):
        undiscovered_tiles = [i for i in range(self.game.N) if not (self.game.is_displayed(i) or i in self.full_graph.nodes)]

        # sample possible mine configurations
        components = [g.nodes for g in self.number_subgraphs]
        configurations = {}
        for subgraph_indices in components:
            # determine union of all adjacent uncleared tiles
            mineable_tiles = list(reduce(lambda x,y: x.union(y), [set([id for id in self.full_graph[i] if not self.full_graph.nodes[id]["flagged"]]) for i in subgraph_indices]))
            
            # determine number of mines to try to apply to this subgraph
            #min_mine_count, max_mine_count = self.determine_min_max_mines(subgraph_indices, mineable_tiles)

            #valid_mine_perms = self.sample_valid_mine_perms(subgraph_indices, mineable_tiles, min_mine_count, max_mine_count, samples)
            valid_mine_perms = self.sample_valid_mine_perms2(subgraph_indices, mineable_tiles, samples)
            if len(valid_mine_perms) > 0:
                configurations[components.index(subgraph_indices)] = valid_mine_perms

        min_mines = sum([min([len(mine_perm) for mine_perm in configs]) for configs in configurations.values()])
        max_mines = sum([max([len(mine_perm) for mine_perm in configs]) for configs in configurations.values()])

        min_residual_mine_count = max(self.game.flags - max_mines, 0)
        max_residual_mine_count = max(self.game.flags - min_mines, 0)

        min_residual_density = min_residual_mine_count / len(undiscovered_tiles) if len(undiscovered_tiles) > 0 else 1
        max_residual_density = max_residual_mine_count / len(undiscovered_tiles) if len(undiscovered_tiles) > 0 else 1
        initial_density = self.game.m / self.game.N

        yolo_cutoff = yolo_cutoff or initial_density

        discovered_tile_probs = {}
        for subgraph_indices, configs in configurations.items():
            counts = {}
            for config in configs:
                for mine in config:
                    if mine in counts.keys():
                        counts[mine] += 1
                    else:
                        counts[mine] = 1
            for k,v in counts.items():
                discovered_tile_probs[k] = v / len(configs)
        
        if len(discovered_tile_probs.items()) > 0:
            min_tile, min_prob = sorted(discovered_tile_probs.items(), key=lambda x: x[1])[0]
            print("Min prob: ", min_prob)
            print("Max residual density: ", max_residual_density)
            if min_prob < yolo_cutoff or max_residual_density < yolo_cutoff:
                # YOLO it
                if min_prob < max_residual_density:
                    self.perform_actions([], [min_tile])
                    return
                random_undiscovered = np.random.choice(undiscovered_tiles)
                self.perform_actions([], [random_undiscovered])
                return
            
        # No YOLO, just flag
        valid_flag_config = False
        while not valid_flag_config:
            tiles_to_flag = []
            if any([x for x in configurations.values() if len(x) > 0]):
                for subgraph_indices, configs in configurations.items():
                    if len(configs) == 0:
                        continue
                    max_tile, _ = sorted([(tile, prob) for tile, prob in discovered_tile_probs.items() if tile in list(reduce(lambda x,y: set(x).union(set(y)), [self.full_graph[i] for i in components[subgraph_indices]]))], key=lambda x: x[1])[-1]
                    configs_with_max_tile = [c for c in configs if max_tile in c]
                    config = configs_with_max_tile[np.random.randint(0, len(configs_with_max_tile))]
                    tiles_to_flag += config
            tiles_to_flag = list(set(tiles_to_flag))
            flag_delta = self.game.flags - len(tiles_to_flag)
            if flag_delta > 0:
                if flag_delta > len(undiscovered_tiles):
                    continue
                random_undiscovered_tiles = list(np.random.choice(undiscovered_tiles, flag_delta, replace=False))
                tiles_to_flag += random_undiscovered_tiles
            valid_flag_config = (len(tiles_to_flag) == self.game.flags)
        
        if len(tiles_to_flag) != len(set(tiles_to_flag)):
            raise Exception("AHHHHHH WTF")

        self.perform_actions(tiles_to_flag, [])
        if self.game.flags != 0:
            raise Exception("i don't even know")
        self.game.clear_unflagged()

    def auto_play(self, max_level, to_completion=False, samples=1e4, yolo_cutoff=None):
        # If the game's already over, bounce
        if self.game.is_complete():
            return
        
        # If the game isn't started, make first move
        if self.game.is_ready():
            self.game.start_game()
            self.game.clear_random_empty()

        while self.game.is_active():
            self.do_deductive_action(max_level=max_level)

            # if the game completed... somethin' went wrong
            if self.game.is_complete():
                break
            
            if not to_completion:
                break
            
            # if we're out of flags, we done!
            if self.game.flags == 0:
                self.game.clear_unflagged()
                break

            # still got flags...
            # determine whether to flag or yolo it and clear a rando
            self.do_probable_actions(samples=samples, yolo_cutoff=yolo_cutoff)
                
        foo = 1