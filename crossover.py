import numpy as np
from pymoo.core.crossover import Crossover

class MOGACCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 2, prob=0.9)  # 2 parents, 2 offspring, prob as float

    def _do(self, problem, X, **kwargs):
        # Get the shape of input X and adjust for correct interpretation
        n_parents, n_mates, n_vars = X.shape
        print(f"X shape: {X.shape}, problem.n_var: {problem.n_var}")

        # Adjust X to ensure n_mates is 1 and n_vars matches problem.n_var
        if n_mates > 1 and n_vars == problem.n_var:
            X = X.reshape(n_parents * n_mates, problem.n_var)  # Flatten if n_mates is incorrect
            n_matings = min(n_parents // 2, (n_parents * n_mates) // 2)  # Calculate matings based on total parents
        else:
            X = X[:, 0, :]  # Take first mate, shape becomes (n_parents, n_vars)
            n_matings = n_parents // 2 if n_parents >= 2 else 1  # Set to at least 1 pair
        
        print(f"Adjusted X shape: {X.shape}, Y shape will be: {(self.n_offsprings, n_matings, problem.n_var)}")

        # Initialize Y with correct shape
        Y = np.zeros((self.n_offsprings, n_matings, problem.n_var), dtype=int)  # Set data type to int
        
        clusters = {}  # Dictionary to simulate clusters based on pe_alloc
        for i in range(X.shape[0]):  # Use adjusted number of parents
            pe_alloc = X[i, :problem.n_pe_types]  # Extract PE allocation for clustering
            cluster_key = tuple(pe_alloc)
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(i)

        mating_count = 0  # Track number of successful matings
        for cluster_key in clusters:
            indices = clusters[cluster_key]
            if len(indices) < 2:
                continue
            for i in range(0, len(indices), 2):  # Pairwise processing
                if mating_count >= n_matings:
                    break  # Stop if maximum matings reached
                parent1_idx = indices[i]
                parent2_idx = indices[i + 1] if i + 1 < len(indices) else indices[0]  # Loop back to first parent if no more pairs
                if parent1_idx >= X.shape[0] or parent2_idx >= X.shape[0]:
                    continue  # Skip invalid indices
                if np.random.random() < float(self.prob):
                    cross_point1 = np.random.randint(0, problem.n_var)
                    cross_point2 = np.random.randint(cross_point1, problem.n_var)
                    # First offspring from parent2 to parent1
                    Y[0, mating_count, :cross_point1] = X[parent1_idx, :cross_point1]
                    Y[0, mating_count, cross_point1:cross_point2] = X[parent2_idx, cross_point1:cross_point2]
                    Y[0, mating_count, cross_point2:] = X[parent1_idx, cross_point2:]
                    # Second offspring from parent1 to parent2
                    Y[1, mating_count, :cross_point1] = X[parent2_idx, :cross_point1]
                    Y[1, mating_count, cross_point1:cross_point2] = X[parent1_idx, cross_point1:cross_point2]
                    Y[1, mating_count, cross_point2:] = X[parent2_idx, cross_point2:]

                    # Constraint enforcement
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
                    mating_count += 1  # Increment mating counter only on successful crossover
        
        # Fill remaining mating slots with parent copies if mating_count < n_matings
        while mating_count < n_matings:
            # Select a random parent to copy
            parent_idx = np.random.randint(0, X.shape[0])
            Y[0, mating_count, :] = X[parent_idx, :]  # Copy parent as first offspring
            Y[1, mating_count, :] = X[parent_idx, :]  # Copy parent as second offspring
            mating_count += 1

        return Y