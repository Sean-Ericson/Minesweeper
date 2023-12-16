#Minesweeper GUI by Sean Ericson

from Minesweeper import *
import MS_AI
import tkinter as tk
import sys

HEADER_SIZE = 35
TILE_SIZE = 20
COLORS = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'pink', 'black']

class MS_App(tk.Frame):
    def __init__(self, root, x, y, mines, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        root.iconbitmap("mine.ico")
        self.root = root

        self.x = x
        self.y = y
        self.mines = mines

        self.game = MS_Game2(x, y, mines)
        self.game.TileDisplayHandlers += [self.handle_tile_displayed]
        self.game.BulkTileDisplayHandlers += [self.handle_bulk_tile_display]
        self.game.TileFlagChangedHandlers += [self.handle_tile_flag_changed]
        self.game.GameResetHandlers += [self.handle_reset]
        self.game.GameCompleteHandlers += [self.handle_game_complete]

        self.header = MS_Header(self)
        self.minefield = MS_Field(self)

        self.header.grid(row=0, column=0, sticky=tk.W+tk.N+tk.E)
        self.minefield.grid(row=1, column=0, sticky=tk.W+tk.S+tk.E)
        root.after(5, self.update)

    def get_game(self):
        return self.game

    def update(self):
        self.minefield.focus_set()
        self.update_time()
        self.root.after(75, self.update)

    def handle_tile_displayed(self, tile_num):
        self.minefield.clear_tile(tile_num)
    
    def handle_bulk_tile_display(self, tiles):
        self.minefield.clear_tiles([tile.id for tile in tiles])

    def handle_tile_flag_changed(self, tile_num):
        if self.game.is_flagged(tile_num):
            self.minefield.place_flag(tile_num)
        else:
            self.minefield.remove_flag(tile_num)
        self.header.update_flag_count(self.game.flags)

    def handle_reset(self):
        self.update_reset()

    def handle_game_complete(self):
        if self.game.is_win():
            self.win_game()
        elif self.game.is_lose():
            self.lose_game()

    def update_time(self):
        if not self.game.is_active():
            return
        time = self.game.get_cur_time()
        self.header.set_time(int(time))

    def button_click(self):
        if self.game.is_active():
            self.game.clear_unflagged()
        else:
            self.game.reset()

    def manual_reset(self):
        self.game.reset()

    def first_move(self):
        if self.game.is_ready():
            self.game.start_game()
        self.game.clear_random_empty()

    def lose_game(self):
        self.header.update_lose()
        self.minefield.reveal_mines(self.game.unflagged_mines())
        self.minefield.reveal_bad_flags(self.game.misplaced_flags())
        return

    def win_game(self):
        self.header.update_win()
        self.minefield.highlight_flags(self.game.get_all_flagged())

    def update_reset(self):
        self.header.update_reset()
        self.minefield.update_reset()
        

class MS_Header(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(bd=5)
        self.root = root

        self.flag_counter = tk.Label(self, text=root.mines)
        self.time_counter = tk.Label(self, text='000')
        self.button = tk.Button(self, text="Clear", command=self.button_callback)

        self.flag_counter.grid(row=0, column=0)
        self.button.grid(row=0, column=1)
        self.time_counter.grid(row=0, column=2)
        self.grid_columnconfigure(1, weight=1)

    def button_callback(self):
        self.root.button_click()
        return

    def set_time(self, time):
        self.time_counter.configure(text="{0:0=3d}".format(time))

    def display_score(self):
        game = self.root.game
        score = game.final_score
        total = game.m
        self.flag_counter.configure(text="{} / {}".format(score, total))

    def update_flag_count(self, flags):
        self.flag_counter.config(text=flags)

    def update_win(self):
        self.button.configure(fg='green', text="Restart")
        self.display_score()

    def update_lose(self):
        self.button.configure(fg='red', text="Restart")
        self.display_score()

    def update_reset(self):
        self.button.configure(fg='black', text="Clear")
        self.update_flag_count(self.root.game.m)
        self.time_counter.configure(text='000')


class MS_Field(tk.Canvas):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(width=root.x*TILE_SIZE+2, height=root.y*TILE_SIZE+2,  highlightthickness=0, borderwidth=0)
        self.root = root

        self.tile_size = TILE_SIZE
        
        self.bind("<1>", self.lclick_callback)
        self.bind("<3>", self.rclick_callback)
        self.bind("<Return>", lambda _: self.root.button_click())
        self.bind("r", lambda _: self.root.manual_reset())
        self.bind("1", lambda _: self.root.first_move())
        self.bind("<Control-Up>", lambda _: self.increment_tile_size())
        self.bind("<Control-Down>", lambda _: self.decrement_tile_size())
        self.new_field()
        return
    
    def increment_tile_size(self):
        self.tile_size += 1
        self.configure(width=self.root.x*self.tile_size+2, height=self.root.y*self.tile_size+2)
        self.new_field(self.root.game.field)
    
    def decrement_tile_size(self):
        self.tile_size -= 1
        self.configure(width=self.root.x*self.tile_size+2, height=self.root.y*self.tile_size+2)
        self.new_field(self.root.game.field)

    def new_field(self, recreate=False):
        if not recreate:
            self.tiles = []
            self.texts = []
        x, y = self.root.x, self.root.y
        for i in range(x*y):
            x1 = (i%x) * self.tile_size
            y1 = (i//x) * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            if not recreate:
                self.tiles.append(self.create_rectangle(x1, y1, x2, y2, outline='black', fill='grey'))
                self.texts.append(self.create_text(((x1+x2)/2, (y1+y2)/2), text=''))
            else:
                self.coords(self.tiles[i], x1, y1, x2, y2)
                self.coords(self.texts[i], ((x1+x2)/2, (y1+y2)/2))


    def update_reset(self):
        self.new_field()

    def click_to_tile(self, event):
        return self.root.game.field.tile_num(event.x // self.tile_size, event.y // self.tile_size)

    def clear_tiles(self, ts):
        game = self.root.game
        for tile in ts:
            num = game.field[tile].mine_count
            if num != 0:
                self.itemconfigure(self.texts[tile], text=str(num), fill=COLORS[num-1])
            self.itemconfigure(self.tiles[tile], fill='white')
        return

    def place_flag(self, t):
        self.itemconfigure(self.texts[t], text='F')
        return 

    def remove_flag(self, t):
        self.itemconfigure(self.texts[t], text='')
        return

    def reveal_mines(self, mines):
        for m in mines:
            self.itemconfigure(self.texts[m], text='M')
            self.itemconfigure(self.tiles[m], fill='red')

    def reveal_bad_flags(self, flags):
        for f in flags:
            self.itemconfigure(self.texts[f], text='!')
            self.itemconfigure(self.tiles[f], fill='yellow')

    def highlight_flags(self, flags):
        for f in flags:
            self.itemconfigure(self.tiles[f], fill='green')

    def lclick_callback(self, event):
        t = self.click_to_tile(event)
        game = self.root.game

        if game.is_complete():
            return

        if (t < 0) or (t > (game.N - 1)):
            raise Exception("bad click?")

        if game.is_ready():
            game.start_game()

        # clear tile
        game.clear(t)

        # Update
        self.root.update()

        return

    def rclick_callback(self, event):
        t = self.click_to_tile(event)
        game = self.root.get_game()

        if game.is_complete():
            return

        if (t < 0) or (t > (game.N - 1)):
            raise Exception("bad click?")
        
        if game.is_ready():
            game.start_game()

        # Flag tile in game
        game.flag(t)

        # Update
        self.root.update()

def main():
    # Get args
    x, y, m = 10, 10, 5
    try:
        x,y,m = sys.argv[1:]
    except:
        pass

    # Create window
    root = tk.Tk()
    root.title("Minesweeper")

    # Start game
    app = MS_App(root, int(x), int(y), mines=int(m), bd=5)
    app.pack()
    root.mainloop()

if __name__ == "__main__":
    main()