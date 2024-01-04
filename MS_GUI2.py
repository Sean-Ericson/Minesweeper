#Minesweeper GUI by Sean Ericson

from Minesweeper import *
from MS_AI2 import MS_AI
import tkinter as tk
import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx

TILE_SIZE = 45
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
        self.game.TileFlagChangedHandlers += [self.handle_tile_flag_changed]
        self.game.GameResetHandlers += [self.handle_reset]
        self.game.GameCompleteHandlers += [self.handle_game_complete]

        self.AI = MS_AI(self.game)

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

    def handle_tile_displayed(self, tiles):
        self.minefield.clear_tiles([tile.id for tile in tiles])

    def handle_tile_flag_changed(self, tile_num):
        if self.game.is_flagged(tile_num):
            self.minefield.place_flag(tile_num)
        else:
            self.minefield.remove_flag(tile_num)
        self.header.update_flag_count(self.game.flags)

    def handle_reset(self):
        self.AI.reset()
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

    def set_thinking(self, thinking):
        self.header.update_thinking(thinking)
        

class MS_Header(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(bd=5)
        self.root = root

        self.flag_counter = tk.Label(self, text=root.mines)
        self.time_counter = tk.Label(self, text='000')
        self.button = tk.Button(self, text="Clear", state="disabled", command=self.button_callback)

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
        score = game.final_score if not any([game.is_mined(i) and game.is_displayed(i) for i in range(game.N)]) else 0
        total = game.m
        self.flag_counter.configure(text="{} / {}".format(score, total))

    def update_flag_count(self, flags):
        self.flag_counter.config(text=flags)
        self.button.configure(state = "disabled" if flags != 0 else "normal")

    def update_win(self):
        self.button.configure(fg='green', text="Restart", state='normal')
        self.display_score()

    def update_lose(self):
        self.button.configure(fg='red', text="Restart", state='normal')
        self.display_score()

    def update_thinking(self, thinking):
        if thinking:
            self.button.configure(text="Thinking...", state=self.button['state'])
        else:
            self.button.configure(fg='black', text="Clear", state=self.button['state'])
        self.update()

    def update_reset(self):
        self.button.configure(fg='black', text="Clear", state='disabled')
        self.update_flag_count(self.root.game.m)
        self.time_counter.configure(text='000')


class MS_Field(tk.Canvas):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(width=root.x*TILE_SIZE+2, height=root.y*TILE_SIZE+2,  highlightthickness=0, borderwidth=0)
        self.root = root

        self.tile_size = TILE_SIZE
        self.display_tile_ids = False
        self.death_tile = None
        
        self.bind("q", lambda ev: print(ev, ev.state, bool(ev.state&4)))
        self.bind("<1>", self.lclick_callback)
        self.bind("<3>", self.rclick_callback)
        self.bind("<Return>", lambda _: self.root.button_click())
        self.bind("r", lambda _: self.root.manual_reset())
        self.bind("s", lambda _: self.root.first_move())
        self.bind("l", lambda _: self.bar())
        self.bind("<Control-g>", lambda _: self.show_full_graph())
        self.bind("<Shift-G>", lambda _: self.show_number_graph())
        self.bind("n", lambda _: self.toggle_tile_ids())
        self.bind("f", lambda _: self.auto_finish())
        self.bind("1", lambda ev: self.full_auto_cheat(1) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(1) if bool(ev.state&4) else self.cheat(1))
        self.bind("2", lambda ev: self.full_auto_cheat(2) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(2) if bool(ev.state&4) else self.cheat(2))
        self.bind("3", lambda ev: self.full_auto_cheat(3) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(3) if bool(ev.state&4) else self.cheat(3))
        self.bind("4", lambda ev: self.full_auto_cheat(4) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(4) if bool(ev.state&4) else self.cheat(4))
        self.bind("5", lambda ev: self.full_auto_cheat(5) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(5) if bool(ev.state&4) else self.cheat(5))
        self.bind("6", lambda ev: self.full_auto_cheat(6) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(6) if bool(ev.state&4) else self.cheat(6))
        self.bind("7", lambda ev: self.full_auto_cheat(7) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(7) if bool(ev.state&4) else self.cheat(7))
        self.bind("8", lambda ev: self.full_auto_cheat(8) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(8) if bool(ev.state&4) else self.cheat(8))
        self.bind("9", lambda ev: self.full_auto_cheat(9) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(9) if bool(ev.state&4) else self.cheat(9))
        self.bind("0", lambda ev: self.full_auto_cheat(10) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(10) if bool(ev.state&4) else self.cheat(10))
        self.bind("m", lambda ev: self.full_auto_cheat(self.root.game.N) if bool(ev.state&131072) and bool(ev.state&4) else self.auto_cheat(self.root.game.N) if bool(ev.state&4) else self.cheat(self.root.game.N))
        self.bind("<Control-Up>", lambda _: self.increment_tile_size())
        self.bind("<Control-Down>", lambda _: self.decrement_tile_size())
        self.new_field()
        return
    
    def bar(self):
        scores = []
        game = self.root.game
        n = 0
        while n<1000:
            n += 1
            self.full_auto_cheat(5)
            score = game.final_score if not game.detonated else 0
            print("n = {}".format(n))
            print("score = {}".format(score))
            scores.append(score)
            print("avg: {}".format(np.mean(scores)))
            print()
            game.reset()
        plt.hist(scores, density=True)
        plt.show()
    
    def increment_tile_size(self):
        self.tile_size += 1
        self.configure(width=self.root.x*self.tile_size+2, height=self.root.y*self.tile_size+2)
        self.new_field(recreate=True)
    
    def decrement_tile_size(self):
        self.tile_size -= 1
        self.configure(width=self.root.x*self.tile_size+2, height=self.root.y*self.tile_size+2)
        self.new_field(recreate=True)

    def toggle_tile_ids(self):
        self.display_tile_ids = not self.display_tile_ids
        for i in range(self.root.x*self.root.y):
            self.itemconfig(self.tile_nums[i], text=(str(i) if self.display_tile_ids else ''))
        self.new_field(recreate=True)

    def new_field(self, recreate=False):
        if not recreate:
            self.tiles = []
            self.texts = []
            self.tile_nums = []
        x, y = self.root.x, self.root.y
        for i in range(x*y):
            x1 = (i%x) * self.tile_size
            y1 = (i//x) * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            if not recreate:
                self.tiles.append(self.create_rectangle(x1, y1, x2, y2, outline='black', fill='grey'))
                self.texts.append(self.create_text(((x1+x2)/2, (y1+y2)/2), text=''))
                self.tile_nums.append(self.create_text((x1, y1), text=(str(i) if self.display_tile_ids else ''), anchor=tk.NW))
            else:
                self.coords(self.tiles[i], x1, y1, x2, y2)
                self.coords(self.texts[i], ((x1+x2)/2, (y1+y2)/2))
                self.coords(self.tile_nums[i], (x1, y1))

    def update_reset(self):
        self.new_field()

    def click_to_tile(self, event):
        return self.root.game.field.tile_num(event.x // self.tile_size, event.y // self.tile_size)

    def clear_tiles(self, ts):
        game = self.root.game
        for tile in ts:
            if self.root.game.is_mined(tile):
                self.death_tile = tile
                return
            num = game.field[tile].mine_count
            if num != 0:
                self.itemconfigure(self.texts[tile], text=str(num), fill=COLORS[num-1])
            self.itemconfigure(self.tiles[tile], fill='white')
        self.update()

    def place_flag(self, t):
        self.itemconfigure(self.texts[t], text='F')
        self.update()

    def remove_flag(self, t):
        self.itemconfigure(self.texts[t], text='')
        self.update()

    def reveal_mines(self, mines):
        for m in mines:
            self.itemconfigure(self.texts[m], text='M')
            self.itemconfigure(self.tiles[m], fill='orange' if self.death_tile is not None and self.death_tile == m else 'red')

    def reveal_bad_flags(self, flags):
        for f in flags:
            self.itemconfigure(self.texts[f], text='!')
            self.itemconfigure(self.tiles[f], fill='yellow')

    def highlight_flags(self, flags):
        for f in flags:
            self.itemconfigure(self.tiles[f], fill='green')

    def cheat(self, n):
        self.root.set_thinking(thinking=True)
        print("Cheat level {} starting".format(n))
        ai = self.root.AI
        game = self.root.game
        if game.is_complete():
            return
        if game.is_ready():
            game.start_game()
            game.clear_random_empty()
        flags, clears = ai.level_n_actions(n)
        for id in flags:
            game.flag(id)
        for id in clears:
            game.clear(id)
        progress = (len(flags) > 0) or (len(clears) > 0)
        print("Cheat complete ({})".format("progress made" if progress else "no progress"))
        self.root.set_thinking(thinking=False)

    def auto_cheat(self, max_level):
        self.root.set_thinking(thinking=True)
        print("Auto-cheat starting (level {})".format(max_level))
        progress_made = self.root.AI.auto_play(max_level)
        print("Auto-cheat complete ({})".format("progress made" if progress_made else "no progress"))
        self.root.set_thinking(thinking=False)

    def full_auto_cheat(self, max_level):
        self.root.set_thinking(thinking=True)
        print("Auto-cheat starting (level {})".format(max_level))
        progress_made = self.root.AI.auto_play(max_level, to_completion=True, samples=2000, yolo_cutoff=0.01)
        print("Auto-cheat complete ({})".format("progress made" if progress_made else "no progress"))
        self.root.set_thinking(thinking=False)

    def auto_finish(self):
        self.root.set_thinking(thinking=True)
        print("Auto-finishing")
        self.root.AI.do_probable_actions(samples=2000, yolo_cutoff=0.075)
        print("Auto-finish complete")
        if not self.root.game.is_complete():
            self.root.set_thinking(thinking=False)

    def show_full_graph(self):
        self.root.AI.display_full_graph()

    def show_number_graph(self):
        self.root.AI.display_number_graph()

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
    #x, y, m = 10, 10, 25
    #x, y, m = 125, 65, 1625
    #x, y, m = 127, 66, 2000
    #x, y, m = 20, 20, 85
    #x, y, m = 40, 40, 320
    x, y, m = 50, 50, 600
    #x, y, m = 16, 30, 99
    #x, y, m = 381, 208, 15850
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