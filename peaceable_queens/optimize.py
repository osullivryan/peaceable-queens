from peaceable_queens.board import create_board
import random
from deap import base, tools, algorithms, creator
import numpy as np
from peaceable_queens import cost
import multiprocessing
import pickle

# Problem dimension
BOARD_SIZE = 16
N_PIECES = 32

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("piece", random.randint, 0, BOARD_SIZE ** 2 - 1)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.piece, N_PIECES * 2
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=BOARD_SIZE ** 2 - 1, indpb=0.10)
toolbox.register("evaluate", cost.cost, n_pieces_each=N_PIECES, board_size=BOARD_SIZE)


def main():
    # Differential evolution parameters
    MU = 300
    N_GEN = 1E10
    print_interv = 100

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

        current_gen += 1
    return pop, hof, logbook


if __name__ == "__main__":
    last_pop, hof, logbook = main()
    print(float(hof[0].fitness.values[0]))
    if hof[0].fitness.values[0] == 0:
        black_pieces = hof[0][:N_PIECES]
        white_pieces = hof[0][N_PIECES:]
        board = create_board(black_pieces, white_pieces, BOARD_SIZE)
        print(board)

        cp = dict(population=last_pop, halloffame=hof,
            logbook=logbook, rndstate=random.getstate())
        with open(f"{BOARD_SIZE}_{N_PIECES}.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
