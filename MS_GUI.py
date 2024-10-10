#Minesweeper GUI by Sean Ericson

from Minesweeper import *
from MS_AI import MS_AI
import tkinter as tk
import pickle
from typing import Optional

class MS_Settings:
    DefaultTileSize = 10
    
    def __init__(self):
        self.tile_size: int = MS_Settings.DefaultTileSize
        self.game = None

class MS_App():
    SETTINGS_FILE = "ms_settings.pk"

    def __init__(self):
        self.game_window = None
        self.settings_window = None
        self.settings = MS_Settings()
        try:
            self.settings = self._get_settings()
        except:
            print("Settings file {} could not be loaded".format(MS_App.SETTINGS_FILE))
            self._save_settings(self.settings)

        self.main_window = tk.Tk()
        self.main_window.resizable(False, False)
        self.main_window.iconbitmap("resources\\mine.ico")
        self.main_window.title("Minesweeper")

        # Menu bar
        menubar = tk.Menu(self.main_window)
        menubar.add_command(label="Settings", command=self._open_settings_window)
        if self.settings.game:
            menubar.add_command(label="Load", command=self._load_saved_game)
        self.main_window.config(menu=menubar)

        # Main Frame
        main_frame = MS_MainFrame(self.main_window, self.settings)
        main_frame.e_StartGameRequest += self._open_game_window
        main_frame.pack()

        self.main_window.mainloop()
    
    def _open_game_window(self, game_args:MS_GameArgs):
        if self.game_window:
            self.game_window.destroy()
        
        # Create window
        self.game_window = MS_GameWindow(self.main_window, game_args, self.settings.tile_size)
        self.game_window.e_SaveGameRequest += self._SaveGameRequest
        self.game_window.mainloop()

    def _load_saved_game(self):
        if self.game_window:
            self.game_window.destroy()
        
        self.settings = self._get_settings()
        args = self.settings.game.args
        # Create window
        self.game_window = MS_GameWindow(self.main_window, args, self.settings.tile_size)
        self.game_window.e_SaveGameRequest += self._SaveGameRequest
        self.game_window.load_game(self.settings.game)
        self.game_window.mainloop()

    def _open_settings_window(self):
        if self.settings_window:
            self.settings_window.destroy()
        self.settings_window = tk.Toplevel(self.main_window)
        self.settings_window.geometry("+{:d}+{:d}".format(self.main_window.winfo_x(), self.main_window.winfo_y()))
        self.settings_window.transient(self.main_window)
        self.settings_window.title("Settings")
        self.settings_window.iconbitmap("resources\\mine.ico")

        settings_frame = MS_SettingsFrame(self.settings_window, self.settings)
        settings_frame.e_SettingsUpdate += self._settings_update
        settings_frame.pack()
        self.settings_window.mainloop()

    def _get_settings(self):
        with open(MS_App.SETTINGS_FILE, 'rb') as f:
            return pickle.load(f)
        
    def _save_settings(self, settings):
        with open(MS_App.SETTINGS_FILE, 'wb') as f:
            pickle.dump(settings, f)

    def _settings_update(self, new_settings):
        self.settings = new_settings
        self._save_settings(self.settings)

    def _SaveGameRequest(self, game):
        self.settings.game = game
        self._save_settings(self.settings)

class MS_MainFrame(tk.Frame):
    def __init__(self, root, settings: MS_Settings):

        self.root = root
        self.settings = settings

        super().__init__(root, bd=10)

        self.e_StartGameRequest = EventSource()

        # Title
        self.title = tk.Label(self, text="Minesweeper", font=("MS Serif", 32))
        self.title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # X/Y/M Labels
        xymFont = ("MS Serif", 14)
        self.xLabel = tk.Label(self, text="Width:", font=xymFont)
        self.yLabel = tk.Label(self, text="Height:", font=xymFont)
        self.mLabel = tk.Label(self, text="Mines:", font=xymFont)
        self.xLabel.grid(row=1, column=0, sticky=tk.E)
        self.yLabel.grid(row=2, column=0, sticky=tk.E)
        self.mLabel.grid(row=3, column=0, sticky=tk.E)

        # X/Y/M Entry
        xyValid = (self.register(lambda s: self._xyValid(s)), '%P')
        mValid = (self.register(lambda s: self._mValid(s)), '%P')
        self.x, self.y, self.m = tk.StringVar(value='10'), tk.StringVar(value='10'), tk.StringVar(value='20')
        self.xEntry = tk.Entry(self, width=10, textvariable=self.x, validate='key', validatecommand=xyValid)
        self.yEntry = tk.Entry(self, width=10, textvariable=self.y, validate='key', validatecommand=xyValid)
        self.mEntry = tk.Entry(self, width=10, textvariable=self.m, validate='key', validatecommand=mValid)
        self.xEntry.grid(row=1, column=1, sticky=tk.W)
        self.yEntry.grid(row=2, column=1, sticky=tk.W)
        self.mEntry.grid(row=3, column=1, sticky=tk.W)
        self.xEntry.bind("<KeyRelease>", self._xyChange)
        self.yEntry.bind("<KeyRelease>", self._xyChange)

        # Start Button
        self.startButton = tk.Button(self, width=12, text="Start", font=("MS Serif", 18), command=lambda: self.e_StartGameRequest.emit(self.game_args()))
        self.startButton.grid(row=4, column=0, columnspan=2, sticky=tk.S, pady=(10, 0))
    
    def _xyValid(self, s: str) -> bool:
        return s == "" or s.isdigit()
    
    def _mValid(self, s):
        return self._xyValid(s) and (0 if s=="" else int(s)) < (self.X() * self.Y())
    
    def _xyChange(self, ev):
        if self.M() >= self.X() * self.Y():
            self.m.set(str(max(self.X() * self.Y() - 1, 1)))
        
    def X(self):
        return int(self.x.get())
    def Y(self):
        return int(self.y.get())
    def M(self):
        return int(self.m.get())
    def game_args(self):
        return MS_GameArgs(self.X(), self.Y(), self.M())

class MS_SettingsFrame(tk.Frame):
    def __init__(self, root, settings: MS_Settings):
        super().__init__(root, bd=10)
        self.pack()
        self.root = root
        self.settings = settings

        self.e_SettingsUpdate = EventSource()

        font = ("MS Serif", 14)
        self.tileSizeLabel = tk.Label(self, text="Tile Size:", font=font)
        self.tileSizeLabel.grid(row=0, column=0, sticky=tk.E)

        tileSizeValid = (self.register(lambda s: self._validate_tile_size(s)), '%P')
        self.tileSize = tk.StringVar()
        self.tileSizeEntry = tk.Entry(self, width=7, textvariable=self.tileSize, validate='key', validatecommand=tileSizeValid)
        self.tileSizeEntry.grid(row=0, column=1, sticky=tk.W)
        self.tileSize.set(str(self.settings.tile_size))

        self.saveButton = tk.Button(self, width=10, text="Save", font=("MS Serif", 14), command=self.save_button_click)
        self.saveButton.grid(row=1, column=0, columnspan=2)

    def _validate_tile_size(self, s):
        return s == "" or s.isdigit()
    
    def save_button_click(self):
        self.settings.tile_size = int(self.tileSize.get())
        self.e_SettingsUpdate.emit(self.settings)
        self.root.destroy()

class MS_GameWindow(tk.Toplevel):
    def __init__(self, root, game_args, tile_size):
        super().__init__(master=root)
        self.root = root
        self.minsize(250, 100)
        self.title("Minesweeper")
        self.iconbitmap("resources\\mine.ico")

        # Menu bar
        self.e_SaveGameRequest = EventSource()
        menubar = tk.Menu(self)
        menubar.add_command(label="Save", command=self._onSaveGameRequest)
        self.config(menu=menubar)

        # Init members
        self.x, self.y, self.mines = game_args.width, game_args.height, game_args.mines
        self.name = None

        # Init game object
        self.game = MS_Game(game_args)
        self._init_game_handlers()

        # Init AI object
        self.AI = MS_AI(self.game)

        # Init window components
        self.header = MS_GameWindow_Header(self)
        self.minefield = MS_GameWindow_Field(self, tile_size=tile_size)

        # Layout window components
        self.header.grid(row=0, column=0, sticky=tk.W+tk.E)
        self.minefield.grid(row=1, column=0)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root.after(5, self.update)

    def _init_game_handlers(self, game=None):
        self.game.e_TilesDisplayed += self._TilesDisplayed
        self.game.e_TileFlagChanged += self._TileFlagChanged
        self.game.e_GameReset += self._GameReset
        self.game.e_GameComplete += self._GameComplete

    def update(self):
        self.minefield.focus_set()
        self.update_time()
        self.root.after(75, self.update)

    def _TilesDisplayed(self, tiles):
        self.minefield.clear_tiles([tile.id for tile in tiles])

    def _TileFlagChanged(self, tile):
        # Update tile
        if tile.flagged:
            self.minefield.place_flag(tile.id)
        else:
            self.minefield.remove_flag(tile.id)
        
        # Update flag counter
        self.header.update_flag_count(self.game.flags)

    def _GameReset(self):
        self.AI.reset()
        self.update_reset()

    def _GameComplete(self):
        if self.game.game_won:
            self.win_game()
        else:
            self.lose_game()

    def load_game(self, game: MS_Game):
        self.game = game
        self._init_game_handlers()
        self.game.start_time = time.time() - self.game.current_time
        self.x, self.y, self.mines = game.x, game.y, game.m
        self.minefield.new_field()
        self.minefield.clear_tiles([t.id for t in game.field.tiles if t.displayed])
        for t in game.flagged_tile_ids():
            self.minefield.place_flag(t)
        self.header.update_flag_count(game.flags)

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
        self.minefield.highlight_flags(self.game.flagged_tile_ids())

    def update_reset(self):
        self.header.update_reset()
        self.minefield.update_reset()

    def set_thinking(self, thinking):
        self.header.update_thinking(thinking)

    def _onSaveGameRequest(self):
        self.game.get_cur_time()
        self.e_SaveGameRequest.emit(game=self.game)

class MS_GameWindow_Header(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(bd=5)
        self.root = root

        self.flag_counter = tk.Label(self, text=root.mines)
        self.time_counter = tk.Label(self, text='000')
        self.button = tk.Button(self, text="Clear", state="disabled", command=self.button_callback)

        self.flag_counter.grid(row=0, column=0, sticky=tk.W)
        self.button.grid(row=0, column=1)
        self.time_counter.grid(row=0, column=2, sticky=tk.E)
        self.grid_columnconfigure(1, weight=1)

    def button_callback(self):
        self.root.button_click()
        return

    def set_time(self, time):
        self.time_counter.configure(text="{0:0=3d}".format(time))

    def display_score(self):
        game = self.root.game
        self.flag_counter.configure(text="{} / {}".format(game.score, game.m))

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

class MS_GameWindow_Field(tk.Canvas):
    NumberColors = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'pink', 'black']
    
    def __init__(self, root, tile_size, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(width=root.x*tile_size+2, height=root.y*tile_size+2,  highlightthickness=0, borderwidth=0)
        self.root = root

        self.tile_size = tile_size
        self.display_tile_ids = False
        self.death_tile = None
        self.number_colors = MS_GameWindow_Field.NumberColors
        
        self.set_bindings()
        self.new_field()
        return
    
    def set_bindings(self):
        self.bind("<1>", self._LeftClick)
        self.bind("<3>", self._RightClick)
        self.bind("<Return>", lambda _: self.root.button_click())
        self.bind("r", lambda _: self.root.manual_reset())
        self.bind("s", lambda _: self.root.first_move())
        self.bind("<Control-g>", lambda _: self.show_full_graph())
        self.bind("<Shift-G>", lambda _: self.show_number_graph())
        self.bind("n", lambda _: self.toggle_tile_ids())
        self.bind("f", lambda _: self.auto_finish())
        self.bind("1", lambda _: self.cheat(1))
        self.bind("2", lambda _: self.cheat(2))
        self.bind("3", lambda _: self.cheat(3))
        self.bind("4", lambda _: self.cheat(4))
        self.bind("5", lambda _: self.cheat(5))
        self.bind("6", lambda _: self.cheat(6))
        self.bind("7", lambda _: self.cheat(7))
        self.bind("8", lambda _: self.cheat(8))
        self.bind("9", lambda _: self.cheat(9))
        self.bind("0", lambda _: self.cheat(10))
        self.bind("<Control-Up>", lambda _: self.increment_tile_size())
        self.bind("<Control-Down>", lambda _: self.decrement_tile_size())
    
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
                self.itemconfigure(self.texts[tile], text=str(num), fill=self.number_colors[num-1])
            self.itemconfigure(self.tiles[tile], fill='white')
        self.update()

    def place_flag(self, t):
        self.itemconfigure(self.texts[t], text='F', fill='red')
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

    def _LeftClick(self, event):
        t = self.click_to_tile(event)
        game = self.root.game

        if game.status == MS_Status.Complete:
            return
        if not 0 <= t < game.N:
            raise Exception("bad click?")

        if game.is_ready():
            game.start_game()

        # check for click on number
        tile = game.field[t]
        if tile.displayed:
            nebs = game.field.get_neighbors(t)
            flag_count = len([neb for neb in nebs if neb.flagged])
            unflagged_nebs = [neb for neb in nebs if not neb.displayed and not neb.flagged]
            if tile.mine_count == flag_count:
                for neb in unflagged_nebs:
                    game.clear(neb.id)

        # clear tile
        game.clear(t)

        # Update
        self.root.update()

        return

    def _RightClick(self, event):
        t = self.click_to_tile(event)
        game = self.root.game

        if game.status == MS_Status.Complete:
            return
        if not 0 <= t < game.N:
            raise Exception("bad click?")

        # Flag tile in game
        game.flag(t)

        # Update
        self.root.update()

def main():
    app = MS_App()

if __name__ == "__main__":
    main()