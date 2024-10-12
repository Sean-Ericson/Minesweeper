#Minesweeper GUI by Sean Ericson
from Minesweeper import *
from MS_AI import MS_AI
from typing import Union, Any
import tkinter as tk
import pickle

from Minesweeper import MS_GameArgs

class MS_Settings:
    """
    Minesweeper game settings to be saved to file.
    """
    DefaultTileSize = 10
    
    def __init__(self) -> None:
        self.tile_size: int = MS_Settings.DefaultTileSize
        self.game: Union[MS_Game, None] = None

class MS_App():
    """
    The Minesweeper application. Just instantiate to run.
    """
    SETTINGS_FILE = "ms_settings.pk"

    def __init__(self) -> None:
        self.game_window: Union[tk.Toplevel, None] = None
        self.settings_window: Union[tk.Toplevel, None] = None
        self.settings: MS_Settings = MS_Settings()
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
    
    def _get_settings(self):
        """
        Open the settings file and return the un-pickled result.
        """
        with open(MS_App.SETTINGS_FILE, 'rb') as f:
            return pickle.load(f)

    def _load_saved_game(self) -> None:
        """
        Load the game saved in the settings object
        """
        # Destroy window if already open
        if self.game_window:
            self.game_window.destroy()
        
        # Get the saved game's args
        self.settings = self._get_settings()
        if self.settings.game is None:
            return
        args = self.settings.game.args

        # Create window
        self.game_window = MS_GameWindow(self.main_window, args, self.settings.tile_size)
        self.game_window.e_SaveGameRequest += self._MS_GameWindow_SaveGameRequest
        self.game_window.load_game(self.settings.game)
        self.game_window.mainloop()
    
    def _MS_GameWindow_SaveGameRequest(self, game: MS_Game) -> None:
        """Handle the game window SaveGameRequest event."""
        self.settings.game = game
        self._save_settings(self.settings)

    def _MS_SettingsFrame_SettingsUpdate(self, new_settings: MS_Settings) -> None:
        """Handle the settings window SettingsUpdate event"""
        self.settings = new_settings
        self._save_settings(self.settings)

    def _open_game_window(self, game_args: MS_GameArgs) -> None:
        """
        Open the game window with the given game arguments.
        """
        # Destroy window if already open
        if self.game_window:
            self.game_window.destroy()
        
        # Create window
        self.game_window = MS_GameWindow(self.main_window, game_args, self.settings.tile_size)
        self.game_window.e_SaveGameRequest += self._MS_GameWindow_SaveGameRequest
        self.game_window.focus_set()
        self.game_window.mainloop()

    def _open_settings_window(self) -> None:
        """Open the settings window."""
        # Destroy window if already open
        if self.settings_window:
            self.settings_window.destroy()
        
        # Create window
        self.settings_window = tk.Toplevel(self.main_window)
        self.settings_window.geometry("+{:d}+{:d}".format(self.main_window.winfo_x(), self.main_window.winfo_y()))
        self.settings_window.transient(self.main_window)
        self.settings_window.title("Settings")
        self.settings_window.iconbitmap("resources\\mine.ico")
        settings_frame = MS_SettingsFrame(self.settings_window, self.settings)
        settings_frame.e_SettingsUpdate += self._MS_SettingsFrame_SettingsUpdate
        settings_frame.pack()
        self.settings_window.mainloop()

    def _save_settings(self, settings: MS_Settings) -> None:
        """Save the given settings to file"""
        with open(MS_App.SETTINGS_FILE, 'wb') as f:
            pickle.dump(settings, f)

class MS_MainFrame(tk.Frame):
    def __init__(self, root: tk.Tk, settings: MS_Settings) -> None:
        super().__init__(root, bd=10)

        # Init members
        self.root: tk.Tk = root
        self.settings: MS_Settings = settings
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
    
    def _mValid(self, s: str) -> bool:
        return self._xyValid(s) and (0 if s=="" else int(s)) < (self.X() * self.Y())
    
    def _xyChange(self, ev) -> None:
        if self.M() >= self.X() * self.Y():
            self.m.set(str(max(self.X() * self.Y() - 1, 1)))
        
    def X(self) -> int:
        return int(self.x.get())
    def Y(self) -> int:
        return int(self.y.get())
    def M(self) -> int:
        return int(self.m.get())
    def game_args(self) -> MS_GameArgs:
        return MS_GameArgs(self.X(), self.Y(), self.M())

class MS_SettingsFrame(tk.Frame):
    def __init__(self, root: tk.Toplevel, settings: MS_Settings):
        super().__init__(root, bd=10)
        
        self.pack()
        self.root: tk.Toplevel = root
        self.settings: MS_Settings = settings
        self.e_SettingsUpdate = EventSource()

        font = ("MS Serif", 14)
        self.tileSizeLabel = tk.Label(self, text="Tile Size:", font=font)
        self.tileSizeLabel.grid(row=0, column=0, sticky=tk.E)

        tileSizeValid = (self.register(lambda s: self._validate_tile_size(s)), '%P')
        self.tileSize = tk.StringVar()
        self.tileSizeEntry = tk.Entry(self, width=7, textvariable=self.tileSize, validate='key', validatecommand=tileSizeValid)
        self.tileSizeEntry.grid(row=0, column=1, sticky=tk.W)
        self.tileSize.set(str(self.settings.tile_size))

        self.saveButton = tk.Button(self, width=10, text="Save", font=("MS Serif", 14), command=self._save_button_click)
        self.saveButton.grid(row=1, column=0, columnspan=2)

    def _save_button_click(self) -> None:
        self.settings.tile_size = int(self.tileSize.get())
        self.e_SettingsUpdate.emit(self.settings)
        self.root.destroy()

    def _validate_tile_size(self, s):
        return s == "" or s.isdigit()
    
class MS_GameWindow(tk.Toplevel):
    def __init__(self, root: tk.Tk, game_args: MS_GameArgs, tile_size: int) -> None:
        super().__init__(master=root)
        self.root: tk.Tk = root
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
        self.header.e_MainButtonClicked += self._MS_GameWindow_Header_MainButtonClicked
        self.field = MS_GameWindow_Field(self, self.x, self.y, tile_size=tile_size)
        self.field.e_TileLeftClick += self._MS_GameWindow_Field_TileLeftClick
        self.field.e_TileRightClick += self._MS_GameWindow_Field_TileRightClick

        # Layout window components
        self.header.grid(row=0, column=0, sticky=tk.W+tk.E)
        self.field.grid(row=1, column=0)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._init_bindings()
        root.after(5, self.update)

    def _init_bindings(self) -> None:
        self.bind("<Return>", lambda _: self._MS_GameWindow_Header_MainButtonClicked())
        self.bind("n", lambda _: self.field.toggle_tile_ids())
        self.bind("r", lambda _: self.manual_reset())
        self.bind("s", lambda _: self.first_move())
        self.bind("<Control-g>", lambda _: self.AI.display_full_graph())
        self.bind("<Shift-G>", lambda _: self.AI.display_full_graph())
        self.bind("<Control-Up>", lambda _: self.field.tile_size_increment())
        self.bind("<Control-Down>", lambda _: self.field.tile_size_decrement())
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

    def _init_game_handlers(self) -> None:
        self.game.e_TilesDisplayed += self._MS_Game_TilesDisplayed
        self.game.e_TileFlagChanged += self._MS_Game_TileFlagChanged
        self.game.e_GameReset += self._MS_Game_GameReset
        self.game.e_GameComplete += self._MS_Game_GameComplete

    def _MS_Game_TilesDisplayed(self, tiles: list[MS_Tile]) -> None:
        self.field.clear_tiles([tile for tile in tiles])

    def _MS_Game_TileFlagChanged(self, tile: MS_Tile) -> None:
        # Update tile
        if tile.flagged:
            self.field.set_tile_flagged(tile.id)
        else:
            self.field.set_tile_unflagged(tile.id)
        
        # Update flag counter
        self.header.update_flag_count(self.game.flags)

    def _MS_Game_GameReset(self) -> None:
        self.AI.reset()
        self._update_reset()

    def _MS_Game_GameComplete(self) -> None:
        if self.game.game_won:
            self.win_game()
        else:
            self.lose_game()

    def _MS_GameWindow_Field_TileLeftClick(self, tile_loc) -> None:
        if self.game.status == MS_Status.Complete:
            return
        if not self.game.field.is_valid_loc(*tile_loc):
            raise Exception("bad click?")

        if self.game.is_ready():
            self.game.start_game()

        # check for click on number
        tile = self.game.field[tile_loc]
        if tile.displayed:
            nebs = self.game.field.neighbors(tile.id)
            flag_count = len([neb for neb in nebs if neb.flagged])
            unflagged_nebs = [neb for neb in nebs if not neb.displayed and not neb.flagged]
            if tile.mine_count == flag_count:
                for neb in unflagged_nebs:
                    self.game.clear_tile(neb.id)

        # clear tile
        self.game.clear_tile(tile.id)

        # Update
        self.update()

    def _MS_GameWindow_Field_TileRightClick(self, tile_loc):
        if self.game.status == MS_Status.Complete:
            return
        if not self.game.field.is_valid_loc(*tile_loc):
            raise Exception("bad click?")

        # Flag tile in game
        self.game.toggle_flag(self.game.field.tile_num(*tile_loc))

        # Update
        self.root.update()

    def _MS_GameWindow_Header_MainButtonClicked(self) -> None:
        if self.game.is_active():
            self.game.clear_unflagged()
        else:
            self.game.reset()

    def _onSaveGameRequest(self) -> None:
        self.game.get_time()
        self.e_SaveGameRequest.emit(game=self.game)

    def _update_reset(self) -> None:
        self.header.update_reset()
        self.field.update_reset()

    def _update_time(self) -> None:
        if not self.game.is_active():
            return
        time = self.game.get_time()
        self.header.set_time(int(time))

    def cheat(self, n) -> None:
        if self.game.is_complete():
            return
        self.set_thinking(thinking=True)
        print("Cheat level {} starting".format(n))
        ai = self.AI
        game = self.game
        if game.is_ready():
            game.start_game()
            game.do_first_move()
        flags, clears = ai.level_n_actions(n)
        for id in flags:
            game.toggle_flag(id)
        for id in clears:
            game.clear_tile(id)
        progress = (len(flags) > 0) or (len(clears) > 0)
        print("Cheat complete ({})".format("progress made" if progress else "no progress"))
        self.set_thinking(thinking=False)

    def first_move(self) -> None:
        if not self.game.is_ready():
            return
        self.game.do_first_move()

    def load_game(self, game: MS_Game) -> None:
        self.game = game
        self._init_game_handlers()
        self.game.start_time = time.time() - self.game.current_time
        self.x, self.y, self.mines = game.x, game.y, game.m
        self.field.new_field()
        self.field.clear_tiles([t for t in game.field if t.displayed])
        for t in game.get_flagged_tile_ids():
            self.field.set_tile_flagged(t)
        self.header.update_flag_count(game.flags)

    def lose_game(self) -> None:
        self.header.update_lose()
        if self.game.detonated_tile:
            self.field.death_tile_num = self.game.detonated_tile.id
        self.field.reveal_mines(self.game.get_unflagged_mine_tile_ids())
        self.field.reveal_bad_flags(self.game.get_misplaced_flag_tile_ids())

    def manual_reset(self) -> None:
        self.game.reset()

    def set_thinking(self, thinking) -> None:
        self.header.update_thinking(thinking)

    def update(self) -> None:
        self._update_time()
        self.root.after(75, self.update)

    def win_game(self) -> None:
        self.header.update_win()
        self.field.highlight_flags(self.game.get_flagged_tile_ids())

class MS_GameWindow_Header(tk.Frame):
    def __init__(self, root: MS_GameWindow, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.configure(bd=5)
        self.root = root
        self.e_MainButtonClicked = EventSource()

        self.flag_counter = tk.Label(self, text=root.mines)
        self.time_counter = tk.Label(self, text='000')
        self.button = tk.Button(self, text="Clear", state="disabled", command=lambda: self.e_MainButtonClicked.emit())

        self.flag_counter.grid(row=0, column=0, sticky=tk.W)
        self.button.grid(row=0, column=1)
        self.time_counter.grid(row=0, column=2, sticky=tk.E)
        self.grid_columnconfigure(1, weight=1)

    def set_time(self, time) -> None:
        self.time_counter.configure(text="{0:0=3d}".format(time))

    def display_score(self) -> None:
        game = self.root.game
        self.flag_counter.configure(text="{} / {}".format(game.score, game.m))

    def update_flag_count(self, flags) -> None:
        self.flag_counter.config(text=flags)
        self.button.configure(state = "disabled" if flags != 0 else "normal")

    def update_win(self) -> None:
        self.button.configure(fg='green', text="Restart", state='normal')
        self.display_score()

    def update_lose(self) -> None:
        self.button.configure(fg='red', text="Restart", state='normal')
        self.display_score()

    def update_thinking(self, thinking) -> None:
        if thinking:
            self.button.configure(text="Thinking...", state=self.button['state'])
        else:
            self.button.configure(fg='black', text="Clear", state=self.button['state'])
        self.update()

    def update_reset(self) -> None:
        self.button.configure(fg='black', text="Clear", state='disabled')
        self.update_flag_count(self.root.game.m)
        self.time_counter.configure(text='000')

class MS_GameWindow_Field(tk.Canvas):
    NumberColors = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'pink', 'black']
    
    def __init__(self, root: MS_GameWindow, x: int, y: int, tile_size: int, *args, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)
        self.configure(width=root.x*tile_size+2, height=root.y*tile_size+2,  highlightthickness=0, borderwidth=0)
        self.root = root

        self.x, self.y = x,y
        self.tile_size = tile_size
        self.display_tile_ids: bool = False
        self.death_tile_num: int = -1
        self.number_colors = MS_GameWindow_Field.NumberColors
        self.tiles: list[int] = []
        self.texts: list[int] = []
        self.tile_nums: list[int] = []
        self.e_TileLeftClick = EventSource()
        self.e_TileRightClick = EventSource()
        
        self.bind("<Button-1>", lambda ev: self.e_TileLeftClick.emit(self._click_to_tile_xy(ev)))
        self.bind("<Button-3>", lambda ev: self.e_TileRightClick.emit(self._click_to_tile_xy(ev)))
        self.new_field()
    
    def _click_to_tile_xy(self, event) -> tuple[int, int]:
        return (event.x // self.tile_size, event.y // self.tile_size)

    def clear_tiles(self, tiles: list[MS_Tile]) -> None:
        for tile in tiles:
            num = tile.mine_count
            if num != 0:
                self.itemconfigure(self.texts[tile.id], text=str(num), fill=self.number_colors[num-1])
            self.itemconfigure(self.tiles[tile.id], fill='white')
        self.update()

    def highlight_flags(self, flagged_tile_nums: list[int]) -> None:
        for f in flagged_tile_nums:
            self.itemconfigure(self.tiles[f], fill='green')

    def new_field(self, recreate: bool = False) -> None:
        for i in range(self.x * self.y):
            x1 = (i % self.x) * self.tile_size
            y1 = (i // self.x) * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            if recreate:
                self.coords(self.tiles[i], x1, y1, x2, y2)
                self.coords(self.texts[i], ((x1+x2)/2, (y1+y2)/2))
                self.coords(self.tile_nums[i], (x1, y1))
            else:
                self.tiles.append(self.create_rectangle(x1, y1, x2, y2, outline='black', fill='grey'))
                self.texts.append(self.create_text(((x1+x2)/2, (y1+y2)/2), text=''))
                self.tile_nums.append(self.create_text((x1, y1), text=(str(i) if self.display_tile_ids else ''), anchor=tk.NW))

    def reveal_bad_flags(self, bad_flag_tile_nums: list[int]) -> None:
        for f in bad_flag_tile_nums:
            self.itemconfigure(self.texts[f], text='!')
            self.itemconfigure(self.tiles[f], fill='yellow')

    def reveal_mines(self, mined_tile_nums: list[int]) -> None:
        for m in mined_tile_nums:
            self.itemconfigure(self.texts[m], text='M')
            self.itemconfigure(self.tiles[m], fill='orange' if self.death_tile_num == m else 'red')

    def set_tile_flagged(self, t) -> None:
        self.itemconfigure(self.texts[t], text='F', fill='red')
        self.update()

    def set_tile_unflagged(self, t) -> None:
        self.itemconfigure(self.texts[t], text='')
        self.update()

    def tile_size_increment(self) -> None:
        self.tile_size += 1
        self.configure(width=self.x*self.tile_size+2, height=self.y*self.tile_size+2)
        self.new_field(recreate=True)
    
    def tile_size_decrement(self) -> None:
        self.tile_size -= 1
        self.configure(width=self.x*self.tile_size+2, height=self.y*self.tile_size+2)
        self.new_field(recreate=True)

    def toggle_tile_ids(self) -> None:
        self.display_tile_ids = not self.display_tile_ids
        for i in range(self.x*self.y):
            self.itemconfig(self.tile_nums[i], text=(str(i) if self.display_tile_ids else ''))
        self.new_field(recreate=True)

    def update_reset(self) -> None:
        self.death_tile_num = -1
        self.new_field()

def main() -> None:
    app = MS_App()

if __name__ == "__main__":
    main()