from peaceable_queens.board import create_board
import random
from deap import base, tools, algorithms, creator
import numpy as np
from peaceable_queens import cost
import multiprocessing
import pickle
import datetime
import os

def main(board_size: int, n_pieces: int):

    checkpoint_dir = 'results/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("piece", random.randint, 0, board_size ** 2 - 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.piece, n_pieces * 2
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=0, up=board_size ** 2 - 1, indpb=0.10
    )
    toolbox.register(
        "evaluate", cost.cost, n_pieces_each=n_pieces, board_size=board_size
    )

    # Differential evolution parameters
    MU = 300
    N_GEN = 1e10
    print_interv = 100
    save_interv = 10000

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    current_gen = 1
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        cxpb=0.5,
        mu=MU,
        lambda_=MU,
        mutpb=0.35,
        ngen=1,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    current_gen += 1
    while float(hof[0].fitness.values[0]) > 0 and current_gen < N_GEN:
        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            cxpb=0.5,
            mu=MU,
            lambda_=MU,
            mutpb=0.35,
            ngen=1,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )
        if current_gen % print_interv == 0:
            print(f"Current generation: {current_gen}")
            print(f"Current best fitness: {float(hof[0].fitness.values[0])}")

        if current_gen % save_interv == 0:
            cp = dict(
                population=pop,
                halloffame=hof,
                logbook=logbook,
                rndstate=random.getstate(),
                stats=stats,
            )
            print('saving')
            with open(f"{checkpoint_dir}/{board_size}_{n_pieces}_{save_interv}.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

        current_gen += 1
    return pop, hof, logbook


if __name__ == "__main__":
    board_size = 16
    n_pieces = 38
    last_pop, hof, logbook = main(board_size, n_pieces)
    print(float(hof[0].fitness.values[0]))
    if hof[0].fitness.values[0] == 0:
        black_pieces = hof[0][:n_pieces]
        white_pieces = hof[0][n_pieces:]
        board = create_board(black_pieces, white_pieces, board_size)
        np.set_printoptions(threshold=np.inf)

        cp = dict(
            population=last_pop,
            halloffame=hof,
            logbook=logbook,
            rndstate=random.getstate(),
            black_pieces=black_pieces,
            white_pieces=white_pieces,
            board=board,
        )
        with open(f"results/{board_size}_{n_pieces}.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
