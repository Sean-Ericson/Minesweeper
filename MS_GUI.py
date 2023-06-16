#Minesweeper GUI by Sean Ericson

import Minesweeper
import MS_AI
import tkinter as tk
import sys
import time

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

        self.game = Minesweeper.MS_Game(x, y, mines)
        self.header = MS_Header(self)
        self.minefield = MS_Field(self)

        self.header.grid(row=0, column=0, sticky=tk.W+tk.N+tk.E)
        self.minefield.grid(row=1, column=0, sticky=tk.W+tk.S+tk.E)
        root.after(5, self.update)

    def get_game(self):
        return self.game

    def game_status(self):
        return self.game.status

    def is_active(self):
        return self.game_status() == 'ACTIVE'

    def update(self):
        self.minefield.focus_set()
        self.update_time()
        if len(self.game.event_queue) > 0:
            type, data = self.game.event_queue.pop(0)
            if type == "is_cleared":
                self.minefield.clear_tiles(data)
            if type == "FLAG_PLACED":
                tile, flags = data
                self.minefield.place_flag(tile)
                self.header.update_flag_count(flags)
            if type == "FLAG_REMOVED":
                tile, flags = data
                self.minefield.remove_flag(tile)
                self.header.update_flag_count(flags)
            if type == "WIN":
                self.win_game()
            if type == "LOSE":
                self.lose_game(data)
            if type == "RESET":
                self.update_reset()
        self.root.after(75, self.update)

    def update_time(self):
        try:
            time = self.game.get_cur_time()
        except:
            return
        self.header.set_time(int(time))

    def button_click(self):
        if self.game.is_active():
            self.game.clear_unflagged()
        else:
            self.game.reset()

    def manual_reset(self, event):
        self.game.reset()

    def flag_cheat(self, event):
        MS_AI.flag_obvious(self.game)

    def clear_cheat(self, event):
        MS_AI.clear_obvious(self.game)

    def super_cheat(self, event):
        MS_AI.AI_1(self.game)

    def first_move(self, event):
        self.game.clear_random_empty()

    def lose_game(self, data):
        mines, flags = data
        self.header.update_lose()
        self.minefield.reveal_mines(mines)
        self.minefield.reveal_bad_flags(flags)
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
        
        self.bind("<1>", self.lclick_callback)
        self.bind("<3>", self.rclick_callback)
        self.bind("<Return>", lambda _: self.root.button_click())
        self.bind("r", self.root.manual_reset)
        self.bind("f", self.root.flag_cheat)
        self.bind("c", self.root.clear_cheat)
        self.bind("1", self.root.first_move)
        self.bind("a", self.root.super_cheat)
        self.new_field()
        return

    def new_field(self):
        self.tiles = []
        self.texts = []
        x, y = self.root.x, self.root.y
        for i in range(x*y):
            x1 = (i%x) * TILE_SIZE
            y1 = (i//x) * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            self.tiles.append(self.create_rectangle(x1, y1, x2, y2, outline='black', fill='grey'))
            self.texts.append(self.create_text(((x1+x2)/2, (y1+y2)/2), text=''))


    def update_reset(self):
        self.new_field()

    def click_to_tile(self, event):
        return self.root.game.tile_num(event.x // TILE_SIZE, event.y // TILE_SIZE)

    def clear_tiles(self, ts):
        game = self.root.game
        for tile in ts:
            num = game.field[tile][1]
            if num != 0:
                self.itemconfigure(self.texts[tile], text=game.field[tile][0], fill=COLORS[num-1])
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
        game = self.root.get_game()

        if (t<0) or (t>game.x*game.y -1):
            raise Exception("bad click?")

        # clear tile
        game.clear(t)

        # Update
        self.root.update()

        return

    def rclick_callback(self, event):
        t = self.click_to_tile(event)
        game = self.root.get_game()

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