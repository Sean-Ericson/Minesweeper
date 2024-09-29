# Minesweeper by Sean Ericson

import random
import time
from enum import Enum

# Class for Minesweeper
class MS_Game:
    def __init__(self, x, y, m):
        if x < 4 or y < 4:
            raise Exception("Field dimensions must be 4x4 or larger.")
        self.x = x
        self.y = y
        self.N = x*y
        if m < 1 or m > self.N - 1:
            raise Exception("Number of mines must be between 1 and x*y - 1, inclusive.")
        self.m = m
        self.flags = m
        self.event_queue = []
        self.status = 'READY'
        self.start_time = 0
        self.total_time = 0
        self.final_score = 0
        self.field = [['?', 0] for _ in range(self.N)]

        # Initialize field
        self.init_field()

        return

    # Return str repr of the field
    def __str__(self):
        s = self.status + '\n'
        for y in range(self.y):
            l = ""
            for x in range(self.x):
                l += self.field[self.tile_num(x,y)][0]
            s += l+'\n'
        return s

    # Place mines and set counts
    def init_field(self):
        # Set mines
        for i in random.sample(list(range(self.N)), self.m):
            self.field[i][1] = -1

        # Set numbers
        for t in range(self.N):
            if self.field[t][1] == -1:
                continue
            nebs = self.neighbors(t)
            mines = len([0 for n in nebs if self.field[n][1] == -1])
            self.field[t][1] = mines

    # Reset the game
    def reset(self):
        self.flags = self.m
        self.final_score = 0
        self.event_queue = [("RESET", None)]
        self.bad_flags = []
        self.status = 'READY'
        self.field = [['?', 0] for _ in range(self.N)]
        self.init_field()

    # Get the elapsed time
    def get_cur_time(self):
        # Current time only valid for active game
        if not self.is_active():
            raise Exception("Called get_cur_time() on inactive game")

        # Return the time
        return time.time() - self.start_time

    # Get the final total time
    def get_total_time(self):
        # Total time only valid for finished games
        if not ((self.status == 'WIN') or (self.status == 'LOSE')):
            raise Exception("Called get_total_time() with invalid game status")
        return self.total_time

    def get_final_score(self):
        if not (self.is_win() or self.is_lose()):
            raise Exception("Called get_final_score() win invalid game status")
        return self.final_score

    # Convert (x,y) to tile number
    def tile_num(self, x, y):
        return y*self.x + x

    # Convert tile number to (x,y)
    def tile_loc(self, n):
        return (n%self.x, n//self.x)

    # Get true value of tile
    def tile_val(self, n):
        return self.field[n][1]

    # Get the display value of a tile
    def tile_disp(self, n):
        return self.field[n][0]

    # Check if (x,y) is a valid tile
    def is_valid_loc(self, x, y):
        return (x>=0) and (x<self.x) and (y>=0) and (y<self.y)

    def is_ready(self):
        return self.status == 'READY'

    # Check if game is active
    def is_active(self):
        return self.status == 'ACTIVE'

    def is_lose(self):
        return self.status == 'LOSE'

    def is_win(self):
        return self.status == 'WIN'

    # Count number of correct flags
    def calc_score(self):
        return len([t for t in self.field if t[0] == 'F' and t[1] == -1])

    # Indicate that the player has lost
    def set_lose(self, data):
        # Set status, update event queue
        self.total_time = time.time() - self.start_time
        self.status = 'LOSE'
        self.final_score = self.calc_score()
        self.event_queue.append(("LOSE", data))

    def set_win(self):
        self.total_time = time.time() - self.start_time
        self.status = 'WIN'
        self.final_score = self.calc_score()
        self.event_queue.append(('WIN', None))

    # Return a list of tile numbers which are adjacent to tile number n
    def neighbors(self, n):
        x,y = self.tile_loc(n)
        nbs = [self.tile_num(x+i, y+j) for i in [-1,0,1] for j in [-1,0,1] if self.is_valid_loc(x+i, y+j)]
        nbs.remove(self.tile_num(x,y))
        return nbs

    # Check if a tile is is_cleared
    def is_cleared(self, n):
        return self.field[n][0] != '?' and self.field[n][0] != 'F'

    def is_number(self, n):
        return self.is_cleared(n) and self.field[n][0] != ''

    # Check if a tile is flagged
    def flagged(self, n):
        return self.field[n][0] == 'F'

    def get_all_flagged(self):
        return [n for n in range(self.N) if self.flagged(n)]

    def is_mine(self, n):
        return self.field[n][1] == -1

    # Return what's displayed on tile n
    def get_tile(self, n):
        return self.field[n][0]

    # Toggle the flagg setting on tile n
    def flag(self, n):
        # Do nothing if game inactive
        if self.status != 'ACTIVE':
            return

        # You can't put a flag there!
        if (n < 0) or (n > self.x*self.y - 1):
            raise Exception("Bad tile in flag()")

        # Get the tile we're workin' with
        tile = self.field[n]

        # Is it unflagged?
        if tile[0] == '?':
            # If we still got spare flags, throw one down
            if self.flags > 0:
                tile[0] = 'F'
                self.flags -= 1
                self.event_queue.append(("FLAG_PLACED", (n, self.flags)))
                    
        # It's flagged already, so remove it
        elif tile[0] == 'F':
            tile[0] = '?'
            self.flags += 1
            self.event_queue.append(("FLAG_REMOVED", (n, self.flags)))

    # Set tile display value to hidden value
    def reveal(self, n):
        tile = self.field[n]
        if tile[1] == -1:
            tile[0] = 'M'
        if tile[1] == 0:
            tile[0] = ''
        else:
            tile[0] = str(tile[1])
        return

    # For the first move, open a random empty tile
    def clear_random_empty(self):
        # only allow on first move
        if self.status != 'READY':
            return
        zero_tiles = []
        empty_tiles = []
        for i in range(self.N):
            if self.field[i][1] == 0:
                zero_tiles.append(i)
            if self.field[i][1] > 0:
                empty_tiles.append(i)
        if len(zero_tiles) > 0:
            self.clear(random.choice(zero_tiles))
        elif len(empty_tiles) > 0:
            self.clear(random.choice(empty_tiles))

    # Try to clear tile n
    def clear(self, n):
        # If the game hasn't started, start it
        if self.status == 'READY':
            self.status = 'ACTIVE'
            self.start_time = time.time()

        # If the game isn't active, do nothing
        if not self.is_active():
            return

        # If you're trying to clear an invalid tile, ur gonna have a bad time.
        if (n < 0) or (n > self.x*self.y - 1):
            raise Exception("Bad tile in clear()")

        # If it's already is_cleared, or it's flagged, do nothin'
        if self.is_cleared(n) or self.flagged(n):
            return

        # If you've is_cleared a mine, bummer
        if self.is_mine(n):
            self.reveal(n)
            self.set_lose(([n], []))
        # Otherwise, cascade clear
        else:
            self.cascade_clear(n)

    # Clear tile n and all adjacent empty tiles, then repeat
    def cascade_clear(self, n):
        stack = [n]
        is_cleared = [n]

        while len(stack) > 0:

            # Take a tile off the stack
            t = stack.pop()
            tile = self.field[t]

            # If the tile is empty and next to no mines
            if tile[1] == 0:

                # Get all its neighbors
                for neb in self.neighbors(t):
                    # For each uncleared neighbor, add to stack
                    if self.field[neb][0] == '?':
                        stack.append(neb)
                        is_cleared.append(neb)

            # Clear the tile
            self.reveal(t)

        # Indicate what was is_cleared
        self.event_queue.append(("is_cleared", is_cleared))

    # Clear all unflagged tiles
    def clear_unflagged(self):
        if self.flags > 0:
            print("noep")
            return

        detonated = []
        bad_flags = []
        is_cleared = []
        for i in range(self.N):
            if self.flagged(i) and not self.is_mine(i):
                self.field[i] = '!'
                bad_flags.append(i)
            if not (self.flagged(i) or self.is_cleared(i)):
                self.reveal(i)
                is_cleared.append(i)
                if self.is_mine(i):
                    detonated.append(i)

        # Indicate clear event
        self.event_queue.append(("is_cleared", is_cleared))

        # Check if the player won
        if len(detonated) == 0:
            self.set_win()
        else:
            self.set_lose((detonated, bad_flags))

class MS_Status(Enum):
    READY = 0
    ACTIVE = 1
    COMPLETE = 2

class MS_Tile:
    def __init__(self, id):
        self.displayed = False
        self.flagged = False
        self.mined = False
        self.mine_count = None
        self.effective_count = None
        self.id = id
        self.neighbors = []
    
    @classmethod
    def from_dict(dict):
        tile = MS_Tile()
        for k, v in dict.items():
            setattr(tile, k, v)
        return tile

    def is_displayed_number(self):
        return self.displayed and self.mine_count > 0

class MS_Field:
    def __init__(self, x, y, m):
        self.x = x
        self.y = y
        self.N = x*y
        self.m = m
        self.tiles = []

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tiles[key]
        elif isinstance(key, tuple):
            return self.tiles[self.tile_num(*key)]
        return None
    
    def __setitem__(self, key, val):
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
            nebs = self.neighbors(i)
            mines = len([n for n in nebs if self.tiles[n].mined])
            self.tiles[i].mine_count = mines
            self.tiles[i].effective_count = mines
            self.tiles[i].neighbors = nebs

    def neighbors(self, n):
        x,y = self.tile_loc(n)
        nbs = [self.tile_num(x+i, y+j) for i in [-1,0,1] for j in [-1,0,1] if self.is_valid_loc(x+i, y+j) and not (i == 0 and j == 0)]
        return nbs
    
    def calc_effective_count(self, n):
        tile = self.tiles[n]
        if tile.mine_count is None:
            return
        if tile.effective_count is None:
            tile.effective_count = tile.mine_count
        tile.effective_count = tile.mine_count - len([x for x in self.neighbors(n) if self.tiles[x].flagged])

    def tile_num(self, x, y):
        return y*self.x + x

    def tile_loc(self, n):
        return (n%self.x, n//self.x)
    

     # Check if (x,y) is a valid tile
    
    def is_valid_loc(self, x, y):
        return (x>=0) and (x<self.x) and (y>=0) and (y<self.y)

class MS_Game2:
    def __init__(self, x, y, m):
        if x < 4 or y < 4:
            raise Exception("Field dimensions must be 4x4 or larger.")
        self.x = x
        self.y = y
        self.N = x*y
        if m < 1 or m > self.N - 1:
            raise Exception("Number of mines must be between 1 and x*y - 1, inclusive.")
        self.m = m
        self.flags = m
        self.status = MS_Status.READY
        self.start_time = 0
        self.detonated = False
        self.game_won = False
        self.game_lost = False
        self.total_time = 0
        self.final_score = 0

        self.TileDisplayHandlers = []
        self.TileFlagChangedHandlers = []
        self.GameCompleteHandlers = []
        self.GameResetHandlers = []

        self.field = MS_Field(x, y, m)

        # Initialize field
        self.init_field()
    
    def call_handlers(self, handlers, *args):
        for handler in handlers:
            handler(*args)

    # Place mines and set counts
    def init_field(self):
        self.field.initialize()

    # Return str repr of the field
    def __str__(self):
        s = self.status + '\n'
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

    def get_tiles(self, tile_ids):
        return [self.field[id] for id in tile_ids]

    # Reset the game
    def reset(self):
        self.flags = self.m
        self.status = MS_Status.READY
        self.start_time = 0
        self.detonated = False
        self.game_won = False
        self.game_lost = False
        self.total_time = 0
        self.final_score = 0
        self.init_field()
        self.call_handlers(self.GameResetHandlers)

    # Get the elapsed time
    def get_cur_time(self):
        # Current time only valid for active game
        if not self.is_active():
            raise Exception("Called get_cur_time() on inactive game")

        # Return the time
        return time.time() - self.start_time

    # Get the final total time
    def get_total_time(self):
        # Total time only valid for finished games
        if self.status != MS_Status.COMPLETE:
            raise Exception("Called get_total_time() with invalid game status")
        return self.total_time

    def get_final_score(self):
        if not (self.is_win() or self.is_lose()):
            raise Exception("Called get_final_score() win invalid game status")
        return self.final_score

    def is_ready(self):
        return self.status == MS_Status.READY

    def is_active(self):
        return self.status == MS_Status.ACTIVE

    def is_complete(self):
        return self.status == MS_Status.COMPLETE
    
    def is_lose(self):
        if self.status != MS_Status.COMPLETE:
            raise Exception("Called is_lose() while game is not complete")
        return self.game_lost

    def is_win(self):
        if self.status != MS_Status.COMPLETE:
            raise Exception("Called is_lose() while game is not complete")
        return self.game_won

    # Count number of correct flags
    def calc_score(self):
        return len([tile for tile in self.field if tile.flagged and tile.mined])

    def start_game(self):
        if self.status != MS_Status.READY:
            raise Exception()
        self.status = MS_Status.ACTIVE
        self.start_time = time.time()

    def set_game_complete(self, game_won):
        self.total_time = time.time() - self.start_time
        self.status = MS_Status.COMPLETE
        self.final_score = self.calc_score()
        if game_won:
            self.game_won = True
        else:
            self.game_lost = True
        self.call_handlers(self.GameCompleteHandlers)

    # Check if a tile is cleared
    def is_displayed(self, n):
        return self.field[n].displayed

    def is_number(self, n):
        return self.field[n].displayed and self.field[n].mine_count > 0

    def is_mined(self, n):
        return self.field[n].mined

    # Check if a tile is flagged
    def is_flagged(self, n):
        return self.field[n].flagged

    def get_all_flagged(self):
        return [n for n in range(self.N) if self.field[n].flagged]

    def unflagged_mines(self):
        return [i for i in range(self.N) if self.is_mined(i) and not self.is_flagged(i)]
    
    def misplaced_flags(self):
        return [i for i in range(self.N) if self.is_flagged(i) and not self.is_mined(i)]

    # Toggle the flagg setting on tile n
    def flag(self, n):
        if self.status != MS_Status.ACTIVE:
            raise Exception("Can only flag while game active.")

        # You can't put a flag there!
        if (n < 0) or (n > self.x*self.y - 1):
            raise Exception("Bad tile in flag()")

        # Get the tile we're workin' with
        tile = self.field[n]

        # Can't flag a displayed tile
        if tile.displayed:
            return

        if tile.flagged:
            self.flags += 1
        else:
            if self.flags > 0:
                self.flags -= 1
            else:
                return

        tile.flagged = not tile.flagged
        for neb in self.field.neighbors(n):
            self.field.calc_effective_count(neb)
        self.call_handlers(self.TileFlagChangedHandlers, (n))

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
        if self.status != MS_Status.ACTIVE:
            raise Exception()

        # If you're trying to clear an invalid tile, ur gonna have a bad time.
        if (n < 0) or (n > self.x*self.y - 1):
            raise Exception("Bad tile in clear()")

        tile = self.field[n]

        # If it's already is_cleared, or it's flagged, do nothin'
        if tile.displayed or tile.flagged:
            return

        # If you've is_cleared a mine, bummer
        if self.is_mined(n):
            self.detonated = True
            self.reveal(n)
            self.call_handlers(self.TileDisplayHandlers, ([self.field[n]]))
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

                # Get all its neighbors
                for neb in tile.neighbors:
                    neb_tile = self.field[neb]
                    # For each uncleared neighbor, add to stack
                    if not neb_tile.displayed:
                        stack.append(neb_tile)
                        
            # Clear the tile
            self.reveal(tile.id)
            if not tile in displayed_tiles:
                displayed_tiles.append(tile)

        # Indicate what was is_cleared
        self.call_handlers(self.TileDisplayHandlers, (displayed_tiles))

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
        self.call_handlers(self.TileDisplayHandlers, (displayed_tiles))