# Minesweeper AIs

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Minesweeper import *

############################################################
# Helper Functions
############################################################


############################################################
# Basic AI Functions
############################################################


############################################################
# AI Version ?
############################################################
        

class MS_AI:
    def __init__(self, game : MS_Game2):
        self.game = game

        game.TileDisplayHandlers += [self.handle_tile_display]
        game.BulkTileDisplayHandlers += [self.handle_bulk_tile_display]
        game.TileFlagChangedHandlers += [self.handle_tile_flag_changed]
        game.GameCompleteHandlers += [self.handle_game_complete]
        game.GameResetHandlers += [self.handle_game_reset]

        self.graph = nx.Graph()

    def handle_tile_display(self, tile):
        if tile.mine_count == 0:
            return
        neighbor_tiles = self.game.get_tiles(tile.neighbors)
        flagged_neighbors = len([n for n in neighbor_tiles if n.flagged])
        self.graph.add_node(tile.id, number=True, value=tile.mine_count, effective_value=tile.mine_count-flagged_neighbors)
        for neighbor in self.game.get_tiles(tile.neighbors):
            if not neighbor.displayed:
                self.graph.add_node(neighbor.id, number=False)
                self.graph.add_edge(tile.id, neighbor.id)

    def handle_bulk_tile_display(self, tiles):
        for tile in tiles:
            self.handle_tile_display(tile)

    def handle_tile_flag_changed(self, tile):
        if not tile.id in self.graph.nodes:
            return
        

    def handle_game_complete(self):
        foo = 1

    def handle_game_reset(self):
        foo = 1