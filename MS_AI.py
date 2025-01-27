# Minesweeper AIs
from math import prod
import networkx as nx
import matplotlib.pyplot as plt
from networkx import connected_components
import numpy as np
from Minesweeper import *
from functools import reduce
from itertools import combinations, chain, product
from networkx.algorithms import bipartite
from typing import Generator
import z3

############################################################
# Helper Functions
############################################################

def get_connected_subgraphs(G: nx.Graph, size: int) -> list[list[int]]:
    # https://stackoverflow.com/a/75873085/14165696
    """Get all connected subgraphs by a recursive procedure"""
    def recursive_local_expand(node_set, possible, excluded, results, max_size) -> None:
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

def all_smt(s: z3.Solver, initial_terms: list[z3.BoolRef]) -> Generator[z3.ModelRef, None, None]:
    """
    Return all satisfying assignments
    (SMT: satisfiability modulo theories)
    """
    def block_term(s: z3.Solver, m: z3.ModelRef, t: z3.BoolRef) -> None:
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s: z3.Solver, m: z3.ModelRef, t: z3.BoolRef) -> None:
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms) -> Generator[z3.ModelRef, None, None]:
        if z3.sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()   
    yield from all_smt_rec(list(initial_terms))

def exactly_n(vars, n: int):
    """
    Return an expression that evaluates to true if exactly n of vars are true
    """
    if not (0 <= n <= len(vars)):
        raise Exception()
    if len(vars) == 1:
        return z3.Not(vars[0]) if n == 0 else vars[0]
    if n == len(vars):
        return reduce(z3.And, vars)
    if n == 0:
        return reduce(z3.And, [z3.Not(v) for v in vars])
    ands = []
    for comb in combinations(vars, n):
        trues = reduce(z3.And, comb) if n > 1 else comb[0]
        others = [z3.Not(var) for var in vars if not var in comb]
        falses = reduce(z3.And, others) if len(others) > 1 else others[0]
        ands.append(z3.And(trues, falses))
    return z3.simplify(reduce(z3.Or, ands))

############################################################
# Helper Classes
############################################################

class FullGraph(nx.Graph):
    """
    Bipartite graph with number tiles making up one type and uncovered tiles the other.
    """
    def __init__(self, field: MS_Field) -> None:
        super().__init__()
        self.field = field

    def tile_displayed(self, tile: MS_Tile) -> None:
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
                for neb_neb in self.field.neighbors(neighbor.id):
                    if neb_neb.displayed:
                        self.add_edge(neb_neb.id, neighbor.id)

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

    def _level_one_actions(self) -> tuple[list[int], list[int]]:
        """
        
        """
        tiles_to_flag = set()
        tiles_to_display = set()

        number_nodes = [id for id,attr in self.full_graph.nodes(data=True) if attr["tile"].displayed]
        for id in number_nodes:
            tile = self.game.field[id]
            unflagged_neighbors = [id for id in self.full_graph[id] if not self.game.field[id].flagged]

            if tile.effective_count == 0:
                tiles_to_display = tiles_to_display.union(unflagged_neighbors)
            elif tile.effective_count == len(unflagged_neighbors):
                tiles_to_flag = tiles_to_flag.union(unflagged_neighbors)

        return list(tiles_to_flag), list(tiles_to_display)

    def _make_number_graph(self) -> nx.Graph:
        number_tiles = []
        for i in self.full_graph.nodes:
            tile = self.full_graph.nodes[i]["tile"]
            if tile.displayed and tile.effective_count > 0:
                number_tiles.append(i)
        graph = nx.Graph(self.full_graph)
        graph.remove_nodes_from([i for i in graph.nodes if graph.nodes[i]["tile"].flagged])
        return bipartite.projected_graph(graph, number_tiles)

    def _MS_Game_TilesDisplayed(self, tiles: list[MS_Tile]) -> None:
        for tile in tiles:
            self.full_graph.tile_displayed(tile)

    def _MS_Game_TileFlagChanged(self, tile: MS_Tile) -> None:
        pass

    def _MS_Game_GameReset(self) -> None:
        self.reset()

    def display_full_graph(self, trimmed: bool=False) -> None:
        full_graph = nx.Graph(self.full_graph)
        if trimmed:
            for id in [tile.id for tile in self.game.field if tile.flagged]:
                if id in full_graph:
                    full_graph.remove_node(id)
            full_graph.remove_nodes_from(list(nx.isolates(full_graph)))
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
    
    def get_valid_mine_assignments(self, number_tiles: list[int], mineable_tiles: list[int]) -> list[list[int]]:
        field = self.game.field
        vars = {i:z3.Bool(f"t{i}") for i in mineable_tiles}
        terms = []
        
        for n in number_tiles:
            tile = self.game.field[n]
            neb_vars = [vars[neb.id] for neb in field.neighbors(n) if neb.id in mineable_tiles]
            terms.append(exactly_n(neb_vars, tile.effective_count))
        
        formula = z3.And(*terms)
        simplified_formula = z3.simplify(formula)
        solver = z3.Solver()
        solver.add(simplified_formula)

        return [[i for i in mineable_tiles if z3.is_true(m[vars[i]])] for m in all_smt(solver, list(vars.values()))]

    def largest_number_subgraph_size(self) -> int:
        number_graph = self._make_number_graph()
        if len(number_graph) == 0:
            return 0
        return max([len(c) for c in nx.connected_components(number_graph)])

    def level_n_actions(self, n: int, nodes=None) -> tuple[list[int], list[int]]:
        if n == 1:
            return self._level_one_actions()

        number_graph = self._make_number_graph()
        tiles_to_display = set()
        tiles_to_flag = set()

        subgraphs = get_connected_subgraphs(number_graph if nodes is None else number_graph.subgraph(nodes), n)
        print(f"\t{len(subgraphs)} subgraphs")
        for subgraph_indices in subgraphs:

            # determine union of all adjacent uncleared tiles
            mineable_tiles = list(reduce(lambda x,y: x.union(y), [set([id for id in self.full_graph[i] if not self.full_graph.nodes[id]["tile"].flagged]) for i in subgraph_indices]))

            valid_perms = self.get_valid_mine_assignments(subgraph_indices, mineable_tiles)
            print(f"{sorted(subgraph_indices)}\t\t{len(valid_perms)} valid assignments")
            if len(valid_perms) == 0:
                continue

            # add to flag/clear lists
            for mineable_tile in mineable_tiles:
                if all([mineable_tile in perm for perm in valid_perms]):
                    tiles_to_flag.add(mineable_tile)
                elif all([not mineable_tile in perm for perm in valid_perms]) and not mineable_tile in tiles_to_display:
                    tiles_to_display.add(mineable_tile)
        
        return list(tiles_to_flag), list(tiles_to_display)    

    def deduction_exhausted_analysis(self):        
        field = self.game.field
        outer_tile_ids = [tile.id for tile in field if not (tile.displayed or tile.id in self.full_graph)]
        number_graph = self._make_number_graph()
        frontier_tiles = set()
        vars = dict()

        # create variables
        for id in self.full_graph:
            tile = field[id]
            if not (tile.displayed or tile.flagged):
                vars[id] = z3.Bool(f"t{id}")
        
        # Get connected components of the number-graph
        connected_components = list(nx.connected_components(number_graph))
        component_asses = {}
        for comp in connected_components:
            mineable_tiles = [t.id for t in reduce(lambda x,y: x.union(y), [set([neb for neb in field.neighbors(tile) if not (neb.displayed or neb.flagged)]) for tile in comp])]
            frontier_tiles = frontier_tiles.union(set(mineable_tiles))
            component_asses[tuple(sorted(comp))] = self.get_valid_mine_assignments(comp, mineable_tiles)

        print("Component assignments: ")
        for k,v in component_asses.items():
            print(k)
            for ass in v:
                print("\t", sorted(ass))

        complete_asses = [a for a in [list(reduce(lambda x,y: set(x).union(set(y)), ass)) for ass in product(*component_asses.values())] if len(a) <= self.game.flags]
        print("complete assignments: ")
        for ass in complete_asses:
            print(sorted(ass))

        frontier_probs = {tile: len([ass for ass in complete_asses if tile in ass]) / len(complete_asses) for tile in frontier_tiles}

        print("Tile mined probs: ")
        for k in sorted(frontier_tiles):
            print(f"{k}: {frontier_probs[k]}")

        print("min frontier prob: ", min(frontier_probs.items(), key=lambda x: x[1]))

        complete_ass_lengths = [len(ass) for ass in complete_asses]
        if len(outer_tile_ids) > 0:
            min_outer_tile_prob = (self.game.flags - max(complete_ass_lengths)) / len(outer_tile_ids)
            max_outer_tile_prob = (self.game.flags - min(complete_ass_lengths)) / len(outer_tile_ids)

            print("min outer prob: ", min_outer_tile_prob)
            print("max outer prob: ", max_outer_tile_prob)

        foo = "bar"
        

    def final_flags(self) -> list[int]:
        field = self.game.field
        outer_tile_ids = [tile.id for tile in field if not (tile.displayed or tile.id in self.full_graph)]
        number_graph = self._make_number_graph()
        frontier_tiles = set()
        vars = dict()

        # create variables
        for id in self.full_graph:
            tile = field[id]
            if not (tile.displayed or tile.flagged):
                vars[id] = z3.Bool(f"t{id}")
        
        # Get connected components of the number-graph
        connected_components = list(nx.connected_components(number_graph))
        component_asses = {}
        for comp in connected_components:
            mineable_tiles = [t.id for t in reduce(lambda x,y: x.union(y), [set([neb for neb in field.neighbors(tile) if not (neb.displayed or neb.flagged)]) for tile in comp])]
            frontier_tiles = frontier_tiles.union(set(mineable_tiles))
            component_asses[tuple(sorted(comp))] = self.get_valid_mine_assignments(comp, mineable_tiles)

        print("Component assignments: ")
        for k,v in component_asses.items():
            print(k)
            for ass in v:
                print("\t", sorted(ass))

        complete_asses = [a for a in [list(reduce(lambda x,y: set(x).union(set(y)), ass)) for ass in product(*component_asses.values())] if len(a) <= self.game.flags]
        print("complete assignments: ")
        for ass in complete_asses:
            print(sorted(ass))

        frontier_probs = {tile: len([ass for ass in complete_asses if tile in ass]) / len(complete_asses) for tile in frontier_tiles}

        print("Tile mined probs: ")
        for k in sorted(frontier_tiles):
            print(f"{k}: {frontier_probs[k]}")

        print("min frontier prob: ", min(frontier_probs.items(), key=lambda x: x[1]))

        complete_ass_lengths = [len(ass) for ass in complete_asses]
        if len(outer_tile_ids) > 0:
            min_outer_tile_prob = (self.game.flags - max(complete_ass_lengths)) / len(outer_tile_ids)
            max_outer_tile_prob = (self.game.flags - min(complete_ass_lengths)) / len(outer_tile_ids)

            print("min outer prob: ", min_outer_tile_prob)
            print("max outer prob: ", max_outer_tile_prob)

        selected_ass = complete_asses[np.random.randint(len(complete_asses))]
        outer_flags = list(np.random.choice(outer_tile_ids, self.game.flags - len(selected_ass)))

        return [int(x) for x in selected_ass + outer_flags]

    def reset(self) -> None:
        self.full_graph = FullGraph(self.game.field)