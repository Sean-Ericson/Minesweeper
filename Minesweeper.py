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
    def __init__(self):
        self.displayed = False
        self.flagged = False
        self.mined = False
        self.number = None

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
        self.game_won = False
        self.game_lost = False
        self.total_time = 0
        self.final_score = 0

        self.TileClearCallback = None
        self.BulkTileClearCallback = None
        self.TileFlaggedCallback = None
        self.GameCompleteCallback = None
        self.GameResetCallback = None

        # Initialize field
        self.init_field()
    
    # Place mines and set counts
    def init_field(self):
        # Set mines
        self.field = [MS_Tile() for _ in range(self.N)]

        for i in random.sample(list(range(self.N)), self.m):
            self.field[i].mined = True

        # Set numbers
        for i in range(self.N):
            if self.field[i].mined:
                continue
            nebs = self.neighbors(i)
            mines = len([n for n in nebs if self.field[n].mined])
            self.field[i].number = mines
    
    def neighbors(self, n):
        x,y = self.tile_loc(n)
        nbs = [self.tile_num(x+i, y+j) for i in [-1,0,1] for j in [-1,0,1] if self.is_valid_loc(x+i, y+j) and not (x == 0 and y == 0)]
        return nbs
    
    def tile_num(self, x, y):
        return y*self.x + x

    # Convert tile number to (x,y)
    def tile_loc(self, n):
        return (n%self.x, n//self.x)

    # Check if (x,y) is a valid tile
    def is_valid_loc(self, x, y):
        return (x>=0) and (x<self.x) and (y>=0) and (y<self.y)

    # Return str repr of the field
    def __str__(self):
        s = self.status + '\n'
        for y in range(self.y):
            l = ""
            for x in range(self.x):
                l += self.field[self.tile_num(x,y)][0]
            s += l+'\n'
        return s

    # Reset the game
    def reset(self):
        self.flags = self.m
        self.status = MS_Status.READY
        self.start_time = 0
        self.game_won = False
        self.game_lost = False
        self.total_time = 0
        self.final_score = 0
        self.init_field()
        if self.GameResetCallback is not None:
            self.GameResetCallback()

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
        self.GameCompleteCallback()

    # Check if a tile is cleared
    def is_displayed(self, n):
        return self.field[n].displayed

    def is_number(self, n):
        return self.field[n].displayed and self.field[n] > 0

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
        self.TileFlaggedCallback(n)

    # Set tile display value to hidden value
    def reveal(self, n):
        self.field[n].displayed = True

    # For the first move, open a random empty tile
    def clear_random_empty(self):
        if self.status != MS_Status.READY:
            raise Exception()

        zero_tiles = []
        number_tiles = []
        for i in range(self.N):
            tile = self.field[i]
            if tile.displayed or tile.flagged or tile.mined:
                continue
            if tile.number == 0:
                zero_tiles.append(i)
            elif tile.number > 0:
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
            self.reveal(n)
            self.set_game_complete(game_won=False)
        # Otherwise, cascade clear
        else:
            self.cascade_clear(n)

    # Clear tile n and all adjacent empty tiles, then repeat
    def cascade_clear(self, n):
        stack = [n]
        displayed_tiles = [n]

        while len(stack) > 0:

            # Take a tile off the stack
            t = stack.pop()
            tile = self.field[t]

            # If the tile is empty and next to no mines
            if tile.number == 0:

                # Get all its neighbors
                for neb in self.neighbors(t):
                    # For each uncleared neighbor, add to stack
                    if not self.field[neb].displayed:
                        stack.append(neb)
                        displayed_tiles.append(neb)

            # Clear the tile
            self.reveal(t)

        # Indicate what was is_cleared
        self.BulkTileClearCallback(displayed_tiles)

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
                    displayed_tiles.append(i)

        # Indicate clear event
        self.BulkTileClearCallback(displayed_tiles)

        self.set_game_complete(game_won=len(detonated)==0)