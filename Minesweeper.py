# Minesweeper by Sean Ericson
import random
import time
from enum import Enum
from typing import Callable, Union
from dataclasses import dataclass

class MS_Status(Enum):
    Ready = 0
    Active = 1
    Complete = 2

@dataclass
class MS_GameArgs:
    width: int = 2
    height: int = 2
    mines: int = 2

class EventSource:
    # https://stackoverflow.com/a/57069782
    def __init__(self):
        self.listeners = []

    def __iadd__(self, listener):
        """Shortcut for using += to add a listener."""
        self.listeners.append(listener)
        return self

    def emit(self, *args, **kwargs):
        for listener in self.listeners:
            listener(*args, **kwargs)

class MS_Tile:
    """
    A tile in a Minesweeper field.
    """
    def __init__(self, id: int):
        self.id = id

        self.displayed: bool = False
        self.flagged: bool = False
        self.mined: bool = False
        self.mine_count: int = 0
        self.effective_count: int = 0

class MS_Field:
    """
    A minefield in a Minesweeper game.
    """
    def __init__(self, x: int, y: int, m: int):
        self.x = x
        self.y = y
        self.N = x*y
        self.m = m
        self.tiles: list[MS_Tile] = []

    def __getitem__(self, key: Union[int, tuple[int, int]]) -> MS_Tile:
        if isinstance(key, int):
            return self.tiles[key]
        elif isinstance(key, tuple):
            return self.tiles[self.tile_num(*key)]
        raise TypeError("Key must be int or tuple of ints.")
    
    def __setitem__(self, key: Union[int, tuple[int, int]], val: MS_Tile):
        if isinstance(key, int):
            self.tiles[key] = val
        elif isinstance(key, tuple):
            self.tiles[self.tile_num(*key)] = val

    def __iter__(self):
        return iter(self.tiles)

    def initialize(self):
        self.tiles = [MS_Tile(i) for i in range(self.N)]
        
        # mines
        for i in random.sample(list(range(self.N)), self.m):
            self.tiles[i].mined = True
        
        # numbers/neighbors
        for i in range(self.N):
            if self.tiles[i].mined:
                continue
            nebs = self.get_neighbors(i)
            mines = len([n for n in nebs if n.mined])
            self.tiles[i].mine_count = mines
            self.tiles[i].effective_count = mines

    def get_neighbors(self, n: int) -> list[MS_Tile]:
        x,y = self.tile_loc(n)
        return [self[x+i, y+j] for i in [-1,0,1] for j in [-1,0,1] if self.is_valid_loc(x+i, y+j) and not (i == 0 and j == 0)]
    
    def calc_effective_count(self, n: int):
        tile = self.tiles[n]
        if tile.mine_count is None:
            return
        if tile.effective_count is None:
            tile.effective_count = tile.mine_count
        tile.effective_count = tile.mine_count - len([x for x in self.get_neighbors(n) if x.flagged])

    def tile_num(self, x: int, y: int) -> int:
        return y*self.x + x

    def tile_loc(self, n: int) -> tuple[int, int]:
        return (n%self.x, n//self.x)
    
    # Check if (x,y) is a valid tile
    def is_valid_loc(self, x: int, y: int) -> bool:
        return (0 <= x < self.x) and (0 <= y < self.y)

class MS_Game:
    """
    A Minesweeper game.
    """
    def __init__(self, args: MS_GameArgs):
        x, y, m = args.width, args.height, args.mines
        if x < 4 or y < 4:
            raise ValueError("Field dimensions must be 4x4 or larger.")
        if not 1 <= m < x*y:
            raise ValueError("Number of mines must be between 1 and x*y - 1, inclusive.")
        
        # Initialize members
        self.x, self.y, self.m = x, y, m
        self.N = x * y
        self.init_members()

        # Initialize field
        self.field = MS_Field(x, y, m)
        self.field.initialize()

        # Set up events
        self.e_TilesDisplayed = EventSource()
        self.e_TileFlagChanged = EventSource()
        self.e_GameComplete = EventSource()
        self.e_GameReset = EventSource()

    def init_members(self):
        self.flags: int = self.m
        self.status: MS_Status = MS_Status.Ready
        self.start_time: float
        self.game_won: bool
        self.score: int
        self.total_time: float

    # Return str repr of the field
    def __str__(self) -> str:
        s = str(self.status) + '\n'
        for y in range(self.y):
            l = ""
            for x in range(self.x):
                tile = self.field[x, y]
                if tile.displayed:
                    if tile.mine_count == 0:
                        l += " "
                    else:
                        l += str(tile.mine_count)
                elif tile.flagged:
                    l += "F"
                else:
                    l += "X"
            s += l+'\n'
        return s

    def get_tiles(self, tile_ids: list[int]) -> list[MS_Tile]:
        return [self.field[id] for id in tile_ids]

    # Reset the game
    def reset(self):
        self.init_members()
        self.field.initialize()
        self.e_GameReset.emit()

    # Get the elapsed time
    def get_cur_time(self) -> float:
        # Current time only valid for active game
        if self.status != MS_Status.Active:
            raise Exception("Game inactive")

        # Return the time
        return time.time() - self.start_time

    # Get the final total time
    def get_total_time(self) -> float:
        # Total time only valid for finished games
        if self.status != MS_Status.Complete:
            raise Exception("Game not complete")
        return self.total_time

    def get_score(self) -> int:
        if self.status != MS_Status.Complete:
            raise Exception("Game not complete")
        return self.score

    def is_ready(self) -> bool:
        return self.status == MS_Status.Ready

    def is_active(self) -> bool:
        return self.status == MS_Status.Active

    def is_complete(self) -> bool:
        return self.status == MS_Status.Complete

    # Count number of correct flags
    def calc_score(self) -> int:
        return len([tile for tile in self.field if tile.flagged and tile.mined])

    def start_game(self):
        if self.status != MS_Status.Ready:
            raise Exception()
        self.status = MS_Status.Active
        self.start_time = time.time()

    def set_game_complete(self, game_won):
        self.total_time = time.time() - self.start_time
        self.status = MS_Status.Complete
        self.score = self.calc_score()
        self.game_won = game_won
        self.e_GameComplete.emit()

    def is_number(self, n) -> bool:
        return self.field[n].displayed and self.field[n].mine_count > 0

    def is_mined(self, n) -> bool:
        return self.field[n].mined

    # Check if a tile is flagged
    def is_flagged(self, n) -> bool:
        return self.field[n].flagged

    def get_all_flagged(self) -> list[int]:
        return [n for n in range(self.N) if self.field[n].flagged]

    def unflagged_mines(self) -> list[int]:
        return [i for i in range(self.N) if self.is_mined(i) and not self.is_flagged(i)]
    
    def misplaced_flags(self) -> list[int]:
        return [i for i in range(self.N) if self.is_flagged(i) and not self.is_mined(i)]

    # Toggle the flag setting on tile n
    def flag(self, n):
        if self.status != MS_Status.Active:
            raise Exception("Game not active")

        # You can't put a flag there!
        if not 0 <= n < self.N:
            raise IndexError()

        # Get the tile we're workin' with
        tile = self.field[n]

        # Can't flag a displayed tile
        if tile.displayed:
            return

        if tile.flagged:
            self.flags += 1 # unflagging tile
        else:
            if self.flags > 0:
                self.flags -= 1 # flagging tile
            else:
                return

        tile.flagged = not tile.flagged # actually change tile flag state
        
        for neb in self.field.get_neighbors(n):
            self.field.calc_effective_count(neb.id)
        self.e_TileFlagChanged.emit(tile)

    # Set tile display value to hidden value
    def reveal(self, n):
        self.field[n].displayed = True

    # For the first move, open a random empty tile
    def clear_random_empty(self):
        zero_tiles = []
        number_tiles = []
        for i in range(self.N):
            tile = self.field[i]
            if tile.displayed or tile.flagged or tile.mined:
                continue
            if tile.mine_count == 0:
                zero_tiles.append(i)
            elif tile.mine_count > 0:
                number_tiles.append(i)
        if len(zero_tiles) > 0:
            self.clear(random.choice(zero_tiles))
        elif len(number_tiles) > 0:
            self.clear(random.choice(number_tiles))

    # Try to clear tile n
    def clear(self, n):
        if self.status != MS_Status.Active:
            raise Exception("Game not active")
        if not 0 <= n < self.N:
            raise IndexError()

        tile = self.field[n]

        # If it's already is_cleared, or it's flagged, do nothin'
        if tile.displayed or tile.flagged:
            return

        # If you've is_cleared a mine, bummer
        if self.is_mined(n):
            self.detonated = True
            self.reveal(n)
            self.e_TilesDisplayed.emit(tiles=[tile])
            self.set_game_complete(game_won=False)
            return
        # Otherwise, cascade clear
        else:
            self.cascade_clear(n)

    # Clear tile n and all adjacent empty tiles, then repeat
    def cascade_clear(self, n):
        stack = [self.field[n]]
        displayed_tiles = []

        while len(stack) > 0:

            # Take a tile off the stack
            tile = stack.pop()

            # If the tile is empty and next to no mines
            if tile.mine_count == 0:

                # Get all uncleared neighbors and add to stack
                neighbors = [neb for neb in self.field.get_neighbors(tile.id) if not neb.displayed]
                for neb in neighbors:
                    stack.append(neb) # For each uncleared neighbor, add to stack
                        
            # Clear the tile
            self.reveal(tile.id)
            if not tile in displayed_tiles:
                displayed_tiles.append(tile)

        # Indicate what was is_cleared
        self.e_TilesDisplayed.emit(tiles=displayed_tiles)

    # Clear all unflagged tiles
    def clear_unflagged(self):
        if self.flags > 0:
            return

        detonated = []
        bad_flags = []
        displayed_tiles = []
        for i in range(self.N):
            tile = self.field[i]
            if tile.displayed:
                continue
            if self.is_flagged(i):
                if not tile.mined:
                    bad_flags.append(i)
            else:
                if self.is_mined(i):
                    detonated.append(i)
                else:
                    displayed_tiles.append(tile)

        self.set_game_complete(game_won=len(detonated)==0)

        # Indicate clear event
        self.e_TilesDisplayed.emit(tiles=displayed_tiles)

    def __getstate__(self):
        print("I'm being pickled")
        vals = self.__dict__
        vals['e_TilesDisplayed'] = []
        vals['e_TileFlageChanged'] = []
        vals['e_GameReset'] = []
        vals['e_GameComplete'] = []
        return vals