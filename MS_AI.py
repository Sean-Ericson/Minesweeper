# Minesweeper AIs

import itertools
import random
from collections import defaultdict

############################################################
# Helper Functions
############################################################

# Return a list of tiles currently displaying numbers
def get_num_tiles(game):
    num_tiles = []
    for i in range(game.N):
        tile = game.field[i]
        try:
            num = int(tile[0])
            num_tiles.append(i)
        except:
            pass
    return num_tiles

def flagged_neighbors(game, tile):
    return [n for n in game.neighbors(tile) if game.flagged(n)]

def unflagged_neighbors(game, tile):
    return [n for n in game.neighbors(tile) if not (game.is_cleared(n) or game.flagged(n))]

def effective_number(game, tile):
    num = game.tile_val(tile)
    if num < 0:
        raise Exception("bad tile in effective_number")
    return num - len(flagged_neighbors(game, tile))

def subsets(s, size):
    return [set(x) for x in list(itertools.combinations(s, size))]

def get_regions(game, num_tiles):
    regions = [None for _ in range(game.N)]
    for num_tile in num_tiles:
        regions[num_tile] = set(unflagged_neighbors(game, num_tile))
    return regions

def get_influencers(num_tiles, regions, num):
    infl = []
    for tile in [x for x in num_tiles if x != num]:
        if len(regions[num].intersection(regions[tile])) > 0:
            infl.append(tile)
    return infl

def get_all_influencers(game, num_tiles, regions):
    infls = [[] for _ in range(game.N)]
    for num_tile in num_tiles:
        infls[num_tile] = get_influencers(num_tiles, regions, num_tile)
    return infls

def restriction_satisfied(restriction, mines):
    min, max, overlap = restriction
    if min > max:
        raise Exception("bad restriction")
    occ = [x for x in mines if x in overlap]
    return min <= len(occ) <= max




############################################################
# Basic AI Functions
############################################################

# TODO: Describe
def flag_obvious(game):
    num_tiles = get_num_tiles(game)

    for t in num_tiles:
        tile = game.field[t]
        num = tile[1]
        nebs = game.neighbors(t)

        flagged_nebs = [n for n in nebs if game.flagged(n)]
        f = len(flagged_nebs)
        if f >= num:
            continue

        unflagged_nebs = [n for n in nebs if not (game.is_cleared(n) or game.flagged(n))]
        uf = len(unflagged_nebs)

        if num - f == uf:
            for n in unflagged_nebs:
                game.flag(n)
    return

# TODO: Describe
def clear_obvious(game):
    cont = True
    while cont:
        tiles = get_num_tiles(game)
        cont = False
        while len(tiles) > 0:
            t = tiles.pop()
            num = game.field[t][1]
            nebs = game.neighbors(t)
            flagged_nebs = [n for n in nebs if game.flagged(n)]
            f = len(flagged_nebs)
            unflagged_nebs = [n for n in nebs if not (game.is_cleared(n) or game.flagged(n))]
            uf = len(unflagged_nebs)
            if (num - f == 0) and (uf > 0):
                for n in unflagged_nebs:
                    game.clear(n)
                cont = True
    return

############################################################
# AI Version 1
############################################################

def cheat(game):

    num_tiles = get_num_tiles(game)
    regions = get_regions(game, num_tiles)
    influencers = get_all_influencers(game, num_tiles, regions)

    safe = set()
    danger = set()

    for num_tile in num_tiles:

        infl = influencers[num_tile]
        regionA = regions[num_tile]
        A_size = len(regionA)
        numA = effective_number(game, num_tile)
        if numA > len(regionA):
            raise Exception("Num greater than region size")
        restrictions = []

        # Determine restrictions
        for i in infl:
            numB = effective_number(game, i)
            regionB = regions[i]
            B_size = len(regionB)
            overlap = regionA.intersection(regionB)
            O_size = len(overlap)
            min_occ = min(numB-len(regionB - overlap), O_size)
            max_occ = max(numB, min(O_size, numA, numB))
            restrictions.append([min_occ, max_occ, overlap])

        assignments = subsets(regionA, numA)
        sat_asses = []
        for ass in assignments:
            valid = True
            x = 1
            for rest in restrictions:
                x += 1
                if not restriction_satisfied(rest, ass):
                    valid = False
                    break
            if valid:
                if not ass in sat_asses:
                    sat_asses.append(ass)

        full_asses = []
        for ass in sat_asses:
            full_ass = dict()
            for tile in regionA:
                full_ass[tile] = (tile in ass)
            full_asses.append(full_ass)

        always_mined = set(regionA)
        always_clear = set(regionA)

        for ass in full_asses:
            for tile, mined in ass.items():
                if mined:
                    always_clear -= {tile}
                else:
                    always_mined -= {tile}

        if len(always_clear.intersection(always_mined)) > 0:
            raise Exception("DAFuq")

        safe = safe.union(always_clear)
        danger = danger.union(always_mined)

    for tile in danger:
        game.flag(tile)
    for tile in safe:
        game.clear(tile)

    return safe, danger

def AI_1(game, verbose=False):
    if game.status == 'READY':
        game.clear_random_empty()
    if not game.status == 'ACTIVE':
        return
    safe = ["tmp entry"]
    danger = []
    while (len(safe) > 0) or (len(danger) > 0):
        if verbose:
            print("\rThinking...{0:0=4d} / {1:}".format(game.flags, game.m), end='')
        safe, danger = cheat(game)

    remaining = [x for x in range(game.N) if not (game.is_cleared(x) or game.flagged(x))]
    guesses = random.sample(remaining, game.flags)
    for g in guesses:
        game.flag(g)
    game.clear_unflagged()
    if verbose:
        print()

############################################################
# AI Version 2
############################################################

class infl_graph:
    def __init__(self, N):
        self.N = N
        self.IDS = set(range(N))
        self.active = set()
        self.edges = dict()

    def add_edge(self, a, b):
        if not (a in self.IDs):
            raise Exception("{} is a bad vertex".format(a))
        if not (b in self.IDs):
            raise Exception("{} is a bad vertex".format(b))

        self.active.add(a)
        self.active.add(b)
        self.edges[a].add(b)
        self.edges[b].add(a)

    def remove_edge(self, a, b):
        if not (a in self.active):
            raise Exception("{} is not active".format(a))
        if not (b in self.active):
            raise Exception("{} is not active".format(b))
        self.edges[a] = self.edges[a] - {b}
        self.edges[b] = self.edges[b] - {a}
        if not self.edges[a]:
            self.active.remove(a)
        if not self.edges[b]:
            self.active.remove(b)

# Tile Types
# ('dead', None)- flagged or zero (or effectively 0)
# ('unknown', None) - no info
# ('number', region) - region is a list of neighboring uncleared unflagged tiles)
# ('frontier', restrictors

# Find every hidden square next to a number
## To be used immediately after first move (flags will screw it up)
def get_frontier(game):
    frontier = set()
    for i in range(game.N):
        if game.is_number(i):
            for n in game.neighbors(i):
                if not game.is_cleared(n):
                    frontier.add(n)
    return frontier

def AI_2(game, verbose=False):

    # Check that the game is ready
    if not game.is_ready():
        game.reset()
    
    # First move
    game.clear_random_empty()

    # Generate infl graph
    graph = infl_graph(game.N)
    front = get_frontier(game)