from pymoo.core.mutation import Mutation
import numpy as np

class MOGACMutation(Mutation):
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = float(prob)

    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)
        for i in range(X.shape[0]):
            if np.random.random() < self.prob:
                # Mutation of allocation strings
                pe_idx = np.random.randint(0, problem.n_pe_types)
                Y[i, pe_idx] = max(0, Y[i, pe_idx] + np.random.choice([-1, 1]))  # اصلاح دسترسی
                # Mutation of task assignment
                task_start = problem.n_pe_types + problem.n_ic_types + problem.n_link_types
                task_end = task_start + problem.n_tasks
                for j in range(task_start, task_end):
                    if np.random.random() < 0.1:
                        Y[i, j] = np.random.randint(0, problem.n_pe_types)
        return Y