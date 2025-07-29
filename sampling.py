import numpy as np
from pymoo.core.sampling import Sampling

class MOGACSampling(Sampling):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _do(self, problem, n_samples, **kwargs):  # main method that call for creating samples by pymoo --> return matrix  --  n_samples = num of solutions that must creat
        n_cores = sum(1 for pe in self.problem.pe_database if pe['type'] == 'core')
        X = np.zeros((n_samples, problem.n_var), dtype=int)  # 2D matrix
        
        for i in range(n_samples):
            pe_alloc = np.random.randint(0, 2, size=self.problem.n_pe_types, dtype=int)
            for task in [t for graph in self.problem.task_graphs for t in graph['tasks']]:
                if not any(self.problem.pe_database[pe]['type'] == 'core' for pe in range(self.problem.n_pe_types) if pe_alloc[pe] > 0):
                    pe_idx = np.random.randint(0, self.problem.n_pe_types)
                    pe_alloc[pe_idx] = 1
            ic_alloc = np.random.randint(0, 2, size=self.problem.n_ic_types, dtype=int) 
            link_alloc = np.random.randint(1, 6, size=self.problem.n_link_types, dtype=int) 
            
            task_assign = np.zeros(self.problem.n_tasks, dtype=int)
            task_offset = 0
            for graph_idx, graph in enumerate(self.problem.task_graphs):
                n_tasks = len(graph['tasks'])
                period = graph['period']
                n_copies = self.problem.hyperperiod // period
                n_real_copies = max(1, int(n_copies * self.problem.real_copy_ratio))
                for copy_idx in range(n_real_copies):
                    for task_id in range(n_tasks):
                        # randomly assign tasks to PEs
                        task_assign[task_offset + copy_idx * n_tasks + task_id] = np.random.randint(0, self.problem.n_pe_types)
                task_offset += n_tasks * n_real_copies
            
            core_assign = np.zeros(n_cores, dtype=int)
            if n_cores > 0:
                core_assign[0] = 0  # Assign to IC0
            
            link_connect = np.zeros(self.problem.n_link_types * self.problem.n_pe_types, dtype=int)
            
            # filling the matrix
            X[i, :self.problem.n_pe_types] = pe_alloc
            X[i, self.problem.n_pe_types:self.problem.n_pe_types + self.problem.n_ic_types] = ic_alloc
            X[i, self.problem.n_pe_types + self.problem.n_ic_types:self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types] = link_alloc
            X[i, self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types:self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types + self.problem.n_tasks] = task_assign
            X[i, self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types + self.problem.n_tasks:self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types + self.problem.n_tasks + n_cores] = core_assign
            X[i, self.problem.n_pe_types + self.problem.n_ic_types + self.problem.n_link_types + self.problem.n_tasks + n_cores:] = link_connect
            
        return X