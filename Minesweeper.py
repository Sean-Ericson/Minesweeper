# Minesweeper by Sean Ericson
import random
import time
from enum import Enum
from typing import Union, Any
from dataclasses import dataclass

class MS_Status(Enum):
    """
    The status of a Minesweeper game:
      Ready: game not yet started
      Active: game started but not yet complete
      Complete: game completed but not yet restarted
    """
    Ready = 0
    Active = 1
    Complete = 2

@dataclass
class MS_GameArgs:
    """
    The arguments to a Minesweeper game: field width/height and number of mines.
    """
    width: int = 4
    height: int = 4
    mines: int = 1

class EventSource:
    """
    Class for event-like functionality.
    See https://stackoverflow.com/a/57069782 
    """
    def __init__(self) -> None:
        self.listeners = []

    def __iadd__(self, listener):
        """Shortcut for using += to add a listener."""
        self.listeners.append(listener)
        return self
    
    def clear(self) -> None:
        """Clear the listener list."""
        self.listeners.clear()

    def emit(self, *args, **kwargs) -> None:
        """Call all listeners with the given args."""
        for listener in self.listeners:
            listener(*args, **kwargs)

class MS_Tile:
    """
    A tile in a Minesweeper field.
    """
    def __init__(self, id: int) -> None:
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
        """
        Create a new set of tiles, randomly set mines then calculate neighbor mine counts.
        """
        # Create list of tiles
        self.tiles = [MS_Tile(i) for i in range(self.N)]
        
        # set mines
        for i in random.sample(list(range(self.N)), self.m):
            self.tiles[i].mined = True
        
        # numbers/neighbors
        for i in range(self.N):
            if self.tiles[i].mined:
                continue
            nebs = self.neighbors(i)
            mines = len([n for n in nebs if n.mined])
            self.tiles[i].mine_count = mines
            self.tiles[i].effective_count = mines

    def is_valid_loc(self, x: int, y: int) -> bool:
        """
        Determine if (x,y) represents a valid tile location.
        """
        return (0 <= x < self.x) and (0 <= y < self.y)

    def neighbors(self, n: int) -> list[MS_Tile]:
        """
        Get a list of neighbor tiles for the given tile number.
        """
        x,y = self.tile_loc(n)
        return [self[x+i, y+j] for i in [-1,0,1] for j in [-1,0,1] if self.is_valid_loc(x+i, y+j) and not (i == 0 and j == 0)]
    
    def tile_loc(self, n: int) -> tuple[int, int]:
        """
        Convert from tile number to (x,y) coords.
        """
        return (n%self.x, n//self.x)

    def tile_num(self, x: int, y: int) -> int:
        """
        Convert from (x,y) coords to tile number.
        """
        return y*self.x + x

class MS_Game:
    """
    A Minesweeper game.
    """
    def __init__(self, args: MS_GameArgs) -> None:
        self.args = args
        x, y, m = args.width, args.height, args.mines
        if x < 4 or y < 4:
            raise ValueError("Field dimensions must be 4x4 or larger.")
        if not 1 <= m < x*y:
            raise ValueError("Number of mines must be between 1 and x*y - 1, inclusive.")
        
        # Initialize members
        self.x, self.y, self.m = x, y, m
        self.N = x * y
        self._init_members()

        # Initialize field
        self.field = MS_Field(x, y, m)
        self.field.initialize()

        # Set up events
        self.e_TilesDisplayed = EventSource()
        self.e_TileFlagChanged = EventSource()
        self.e_GameComplete = EventSource()
        self.e_GameReset = EventSource()

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

    def __getstate__(self) -> dict[str, Any]:
        print("I'm being pickled")
        vals = self.__dict__
        vals['e_TilesDisplayed'].clear()
        vals['e_TileFlagChanged'].clear()
        vals['e_GameReset'].clear()
        vals['e_GameComplete'].clear()
        return vals

    def _cascade_clear(self, n: int) -> None:
        """
        Reveals tile number n, and all adjacent tiles if the given tile's neighbor mine count is 0. \n
        This process is repeated for any adjacent tiles if their neighbor mine count is also 0. \n
        Raises the TilesDisplayed event with a list of the tiles displayed.
        """
        stack = [self.field[n]]
        displayed_tiles = []

        while len(stack) > 0:

            # Take a tile off the stack
            tile = stack.pop()

            # If the tile is empty and next to no mines
            if tile.mine_count == 0:
                for neb in self.field.neighbors(tile.id):
                    if not (neb.displayed or neb in stack):
                        stack.append(neb) # For each uncleared neighbor, add to stack
                        
            # Clear the tile
            tile.displayed = True
            if not tile in displayed_tiles:
                displayed_tiles.append(tile)

        # Indicate what was is_cleared
        self.e_TilesDisplayed.emit(tiles=displayed_tiles)

    def _init_members(self) -> None:
        """
        Initialize class members.
        """
        self.flags: int = self.m
        self.status: MS_Status = MS_Status.Ready
        self.start_time: float = 0
        self.game_won: bool = False
        self.score: int = 0
        self.final_time: float = 0
        self.current_time: float = 0
        self.detonated_tile: Union[MS_Tile, None] = None

    def _reveal_tile(self, n) -> None:
        """
        Reveal tile number n. Raise the TilesDisplayed event with a list containing the tile.
        """
        tile = self.field[n]
        tile.displayed = True
        self.e_TilesDisplayed.emit(tiles=[tile])

    def _set_game_complete(self, game_won, score) -> None:
        """
        Set game status to complete, calculate total time.
        Save game_won and score.
        Raise GameComplete event.
        """
        self.final_time = time.time() - self.start_time
        self.status = MS_Status.Complete
        self.score = score
        self.game_won = game_won
        self.e_GameComplete.emit()

    def clear_tile(self, n) -> None:
        """
        Reveal tile number n.
        """
        if self.status != MS_Status.Active:
            raise Exception("Game not active")
        if not 0 <= n < self.N:
            raise IndexError()

        tile = self.field[n]

        # If it's already is_cleared, or it's flagged, do nothin'
        if tile.displayed or tile.flagged:
            return

        # If you've is_cleared a mine, bummer
        if tile.mined:
            self.detonated = True
            self.detonated_tile = tile
            self._reveal_tile(n)
            self._set_game_complete(game_won=False, score=0)
            return
        # Otherwise, cascade clear
        else:
            self._cascade_clear(n)

    def clear_unflagged(self) -> None:
        """
        Clear all unflagged tiles. Only allowed when all flags are placed. \n
        Set game complete with game won iff all flags placed correctly and score given by number of correctly placed flags. \n
        Raise TilesDisplayed event with a list of the displayed tiles.
        """
        if self.flags > 0:
            return

        detonated = []
        bad_flags = []
        displayed_tiles = []
        for i in range(self.N):
            tile = self.field[i]
            if tile.displayed:
                continue
            if tile.flagged:
                if not tile.mined:
                    bad_flags.append(i)
            else:
                if tile.mined:
                    detonated.append(i)
                else:
                    displayed_tiles.append(tile)

        game_won = len(detonated) == 0
        score = len([tile for tile in self.field if tile.flagged and tile.mined])
        self._set_game_complete(game_won, score)

        # Indicate clear event
        self.e_TilesDisplayed.emit(tiles=displayed_tiles)

    def do_first_move(self) -> None:
        """
        Randomly reveal an un-mined tile as the first move of the game.
        Selects a tile with no neighbor mines if possible.
        """
        if self.status != MS_Status.Ready:
            raise Exception("Game must not be Active or Complete.")
        self.start_game()
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
            self.clear_tile(random.choice(zero_tiles))
        elif len(number_tiles) > 0:
            self.clear_tile(random.choice(number_tiles))

    def get_flagged_tile_ids(self) -> list[int]:
        """Get the IDs of all tiles that are currently flagged."""
        return [n for n in range(self.N) if self.field[n].flagged]
    
    def get_misplaced_flag_tile_ids(self) -> list[int]:
        """Get the IDs of all tiles that are flagged but do not have a mine."""
        return [tile.id for tile in self.field if tile.flagged and not tile.mined]

    def get_time(self) -> float:
        """
        If the game is active, compute and return the current time. 
        Otherwise, return the final time.
        """
        # Current time only valid for active game
        if self.status == MS_Status.Ready:
            raise Exception("Game not run or already reset.")

        # Return the time
        if self.status == MS_Status.Active:
            self.current_time = time.time() - self.start_time
            return self.current_time
        
        return self.final_time
    
    def get_unflagged_mine_tile_ids(self) -> list[int]:
        """Get the IDs of all tiles that have mines but are unflagged."""
        return [tile.id for tile in self.field if tile.mined and not tile.flagged]

    def is_active(self) -> bool:
        """True iff the game status is Active."""
        return self.status == MS_Status.Active

    def is_complete(self) -> bool:
        """True iff the game status is Complete."""
        return self.status == MS_Status.Complete

    def is_ready(self) -> bool:
        """True iff the game status is Ready."""
        return self.status == MS_Status.Ready

    def reset(self) -> None:
        """
        Reset the game. Raise the GameReset event.
        """
        self._init_members()
        self.field.initialize()
        self.e_GameReset.emit()

    def start_game(self) -> None:
        """Start the game: set status to active and record initial time."""
        if self.status != MS_Status.Ready:
            raise Exception()
        self.status = MS_Status.Active
        self.start_time = time.time()

    def toggle_flag(self, n) -> None:
        """
        Toggle the flagged status of tile number n.
        Raise the TileFlagChanged event.
        """
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

        # Can't flag if we're out of flags
        if not tile.flagged and self.flags == 0:
            return 
        
        # Change flag count
        self.flags += 1 if tile.flagged else -1

        # Actually change tile flag state
        tile.flagged = not tile.flagged
        
        # Update effective count of neighbors
        for neb in self.field.neighbors(n):
            neb.effective_count += 1 if tile.flagged else -1 
        
        # Raise event
        self.e_TileFlagChanged.emit(tile)