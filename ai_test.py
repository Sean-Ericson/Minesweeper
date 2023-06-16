# Test MS AIs
# By Hadamard_CX

import Minesweeper
import numpy as np
import matplotlib.pyplot as plt

N = 200
P = 25
dims = [(x,x) for x in range(2, 41)]

def run_tests(game, ai, num):
    scores = []
    times = []

    for _ in range(num):
        game.reset()
        ai(game)
        if not (game.is_lose() or game.is_win()):
            raise Exception("AI did not complete game")
        scores.append(game.get_final_score())
        times.append(game.get_total_time())
    return scores, times


def make_ave_plot(x, y, N, filename, disp=False, verbose=False):
    xs = [int(i) for i in np.linspace(1, x*y-1, P)]
    xs.sort()
    aves = []
    for i in xs:
        if verbose:
            print("\r{} / {}           ".format(xs.index(i)+1, len(xs)), end='')
        game = Minesweeper.MS_Game(x, y, i)
        scores, _ = run_tests(game, Minesweeper.ms_ai, N)
        scores = [s / i for s in scores]
        aves.append(np.mean(scores))
    print()
    xs = [i / (x*y) for i in xs]

    plt.clf()
    plt.plot(xs, aves)
    plt.plot(xs, xs)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.title("Mean Score - {}x{} Board".format(x, y))
    plt.xlabel("Mines")
    plt.ylabel("Percent Cleared")
    plt.savefig(filename)
    if disp:
        plt.show()

def make_time_plot(x, y, N, filename, disp=False, verbose=False):
    xs = [int(i) for i in np.linspace(1, x*y-1, P)]
    aves = []
    for i in xs:
        if verbose:
            print("\r{} / {}           ".format(xs.index(i)+1, len(xs)), end='')
        game = Minesweeper.MS_Game(x, y, i)
        _, times = run_tests(game, Minesweeper.ms_ai, N)
        times = [t*1000 for t in times]
        aves.append(np.mean(times))
    print()
    xs = [i / (x*y) for i in xs]

    plt.clf()
    plt.plot(xs, aves)
    plt.title("Mean Time - {}x{} Board".format(x, y))
    plt.xlabel("Percent Mined")
    plt.ylabel("Time (ms)")
    plt.savefig(filename)
    if disp:
        plt.show()

def make_time_multiplot(dims, N, filename, disp=False, verbose=False):
    xdata = []
    tdata = []
    for dim in dims:
        x,y = dim
        xs = [int(i) for i in np.linspace(1, x*y-1, P)]
        aves = []
        for i in xs:
            if verbose:
                print("\r\rDim {} / {}    Mine {} / {}             ".format(dims.index(dim)+1, len(dims), xs.index(i), len(xs)), end='')
            game = Minesweeper.MS_Game(x, y, i)
            _, times = run_tests(game, Minesweeper.ms_ai, N)
            times = [t*1000 for t in times]
            aves.append(np.mean(times))
        xdata.append([i / (x*y) for i in xs])
        tdata.append(aves)
    
    plt.clf()
    for xs, ts, dim in zip(xdata, tdata, dims):
        x, y = dim
        plt.plot(xs, ts, label="{}x{}".format(x, y))
    plt.title("Mean Times")
    plt.xlabel("Percent Mined")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()

make_time_multiplot([(i, i) for i in range(5, 15)], N, "test.png", disp=True, verbose=True)
