import numpy as np
from pymoo.core.crossover import Crossover

class MOGACCrossover(Crossover):
    def __init__(self, prob=0.9):
        # Explicitly set prob as a float
        self.prob = float(prob)
        # Initialize parent class with 2 parents and 2 offspring
        super().__init__(n_parents=2, n_offsprings=2, prob=self.prob)

    def _do(self, problem, X, **kwargs):
        # Get the shape of input X: (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        # Get pop_size from algorithm (passed via kwargs)
        algorithm = kwargs.get('algorithm')
        pop_size = algorithm.pop_size if algorithm is not None else 100  # Default to 100 if not available
        # Calculate number of matings based on desired offspring
        n_matings = min(n_matings, pop_size // self.n_offsprings)
        # Initialize output Y with shape (n_offsprings, n_matings, n_var)
        Y = np.zeros((self.n_offsprings, n_matings, n_var), dtype=int)

        # Cluster parents based on pe_alloc for mating selection
        clusters = {}
        for i in range(n_matings):
            for p in range(n_parents):
                pe_alloc = X[p, i, :problem.n_pe_types]
                cluster_key = tuple(pe_alloc)
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append((p, i))

        mating_count = 0
        for cluster_key in clusters:
            indices = clusters[cluster_key]
            np.random.shuffle(indices)  # Shuffle for random pairing
            for i in range(0, len(indices), 2):
                if mating_count >= n_matings:
                    break
                if i + 1 >= len(indices):
                    # If only one parent in cluster, copy it
                    p1, m1 = indices[i]
                    Y[0, mating_count, :] = X[p1, m1, :]
                    Y[1, mating_count, :] = X[p1, m1, :]
                    mating_count += 1
                    continue
                # Select parent pair
                p1, m1 = indices[i]
                p2, m2 = indices[i + 1]
                if np.random.random() < self.prob.value:  # Use self.prob.value instead of float(self.prob)
                    # Perform two-point crossover
                    cross_point1 = np.random.randint(0, n_var)
                    cross_point2 = np.random.randint(cross_point1, n_var)
                    # First offspring
                    Y[0, mating_count, :cross_point1] = X[p1, m1, :cross_point1]
                    Y[0, mating_count, cross_point1:cross_point2] = X[p2, m2, cross_point1:cross_point2]
                    Y[0, mating_count, cross_point2:] = X[p1, m1, cross_point2:]
                    # Second offspring
                    Y[1, mating_count, :cross_point1] = X[p2, m2, :cross_point1]
                    Y[1, mating_count, cross_point1:cross_point2] = X[p1, m1, cross_point1:cross_point2]
                    Y[1, mating_count, cross_point2:] = X[p2, m2, cross_point2:]

                    # Enforce constraints
                    task_start = problem.n_pe_types + problem.n_ic_types + problem.n_link_types
                    task_end = task_start + problem.n_tasks
                    for j in range(task_start, task_end):
                        Y[0, mating_count, j] = min(int(Y[0, mating_count, j]), problem.n_pe_types - 1)
                        Y[1, mating_count, j] = min(int(Y[1, mating_count, j]), problem.n_pe_types - 1)
                    n_cores = sum(1 for pe in problem.pe_database if pe['type'] == 'core')
                    core_start = task_end
                    core_end = core_start + n_cores
                    for j in range(core_start, core_end):
                        Y[0, mating_count, j] = min(int(Y[0, mating_count, j]), problem.n_ic_types - 1)
                        Y[1, mating_count, j] = min(int(Y[1, mating_count, j]), problem.n_ic_types - 1)
                else:
                    # Copy parents directly if no crossover
                    Y[0, mating_count, :] = X[p1, m1, :]
                    Y[1, mating_count, :] = X[p2, m2, :]
                mating_count += 1

        # Fill remaining matings with random parent copies
        while mating_count < n_matings:
            # Select random parents
            p1 = np.random.randint(0, n_parents)
            m1 = np.random.randint(0, n_matings)
            Y[0, mating_count, :] = X[p1, m1, :]
            Y[1, mating_count, :] = X[p1, m1, :]  # Copy same parent for consistency
            mating_count += 1

        return Y