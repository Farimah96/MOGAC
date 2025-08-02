import numpy as np
from pymoo.core.problem import ElementwiseProblem
from math import lcm

class Problem(ElementwiseProblem):
    def __init__(self, n_pe_types, n_ic_types, n_link_types, task_graphs, pe_database, ic_database, link_database, real_copy_ratio=0.2):
        self.task_graphs = task_graphs
        self.hyperperiod = lcm(*[graph['period'] for graph in task_graphs])  # * = Unpacking Operator  -- graph['period'] = comprehension list
        self.real_copy_ratio = real_copy_ratio
        n_tasks = 0
        for graph in task_graphs:
            n_copies = self.hyperperiod // graph['period']
            n_real_copies = max(1, int(n_copies * real_copy_ratio))
            n_tasks += len(graph['tasks']) * n_real_copies
        
        n_cores = sum(1 for pe in pe_database if pe['type'] == 'core') #generator expression -> "if" = true -> generate 1 --> n_cores in this database equals 3
        n_var = n_pe_types + n_ic_types + n_link_types + n_tasks + n_cores + (n_link_types * n_pe_types)
        
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=np.array([10] * n_pe_types + [10] * n_ic_types + [10] * n_link_types + [n_pe_types - 1] * n_tasks + [n_ic_types - 1] * n_cores + [n_pe_types - 1] * (n_link_types * n_pe_types)),
            type_var=int
        )
        self.n_pe_types = n_pe_types
        self.n_ic_types = n_ic_types
        self.n_link_types = n_link_types
        self.n_tasks = n_tasks
        self.pe_database = pe_database
        self.ic_database = ic_database
        self.link_database = link_database

    def _calculate_slack(self, graph_idx, task_id, current_time):
        deadline = self.task_graphs[graph_idx]['tasks'][task_id].get('deadline', float('inf'))
        return deadline - current_time.get(0, 0)

    def _calculate_comm_time(self, packets, src_pe, tgt_pe):
        if src_pe != tgt_pe:
            return packets * 3
        return 0

    def _schedule_tasks(self, task_assign, link_alloc, link_connect, core_assign):
        schedule = []
        link_usage = 0
        processed_edges = set()
        
        task_offset = 0
        for graph_idx, graph in enumerate(self.task_graphs):
            period = graph['period']
            n_tasks = len(graph['tasks'])
            n_copies = self.hyperperiod // period
            n_real_copies = max(1, int(n_copies * self.real_copy_ratio))
            
            for copy_idx in range(n_real_copies):  # real copies scheduling
                start_offset = copy_idx * period
                copy_schedule = [None] * n_tasks
                current_time = {pe_id: start_offset for pe_id in range(self.n_pe_types)}
                dependencies = [0] * n_tasks
                ready_tasks = []
                completed_tasks = set()
                
                for edge in graph['edges']:
                    _, tgt, _ = edge  #  src , tgt, packet size
                    dependencies[tgt] += 1
                
                for task_id in range(n_tasks): 
                    if dependencies[task_id] == 0:   # tasks that have not dependency
                        slack = self._calculate_slack(graph_idx, task_id, current_time)
                        ready_tasks.append((task_id, slack))
                ready_tasks.sort(key=lambda x: x[1])
                
                while ready_tasks:  # task scheduling loop
                    task_id, _ = ready_tasks.pop(0)
                    pe_id = int(task_assign[task_offset + copy_idx * n_tasks + task_id])
                    duration = graph['tasks'][task_id]['duration']
                    max_dependency_end = start_offset
                    comm_time = 0 
                    
                    for edge in graph['edges']:
                        src, tgt, packets = edge
                        if tgt == task_id and src in completed_tasks:
                            src_pe = int(task_assign[task_offset + copy_idx * n_tasks + src])
                            src_end_time = copy_schedule[src][1]
                            comm_time = self._calculate_comm_time(packets, src_pe, pe_id)
                            max_dependency_end = max(max_dependency_end, src_end_time + comm_time)
                            # Check link connection and adjust communication time if incompatible
                            link_idx = link_alloc[0] - 1 if link_alloc[0] > 0 else 0
                            connect_pe = link_connect[link_idx * self.n_pe_types + pe_id] if link_idx < len(link_connect) // self.n_pe_types else pe_id
                            if connect_pe != src_pe and connect_pe >= 0:
                                additional_delay = 2  # Additional delay for incompatible connection
                                comm_time += additional_delay
                            # Adjust scheduling based on core assignment
                            core_idx = core_assign[pe_id] if pe_id < len(core_assign) else 0
                            if core_idx > 0:
                                core_delay = graph['tasks'][task_id]['duration'] * 0.1  # 10% delay for additional core usage
                                max_dependency_end += core_delay
                            # Update link usage only once per edge
                            if link_alloc[0] > 0 and (graph_idx, copy_idx, src, tgt) not in processed_edges:
                                link_usage += packets
                                processed_edges.add((graph_idx, copy_idx, src, tgt))
                                print(f"Edge ({src}, {tgt}) in graph {graph_idx}, copy {copy_idx}: {packets} packets, comm_time={comm_time}")
                                                                                                        
                    start_time = max(max_dependency_end, start_offset + comm_time)
                    if graph_idx == 0 and task_id == 2 and copy_idx == 0:
                        start_time = max(start_time, 12)  ############## why 12? -> special constraint for 2nd task in 1st graph
                    end_time = start_time + duration
                    copy_schedule[task_id] = [start_time, end_time]
                    current_time[pe_id] = max(current_time[pe_id], end_time)
                    completed_tasks.add(task_id)
                    
                    for edge in graph['edges']:
                        src, tgt, _ = edge
                        if src == task_id:
                            dependencies[tgt] -= 1
                            if dependencies[tgt] == 0 and tgt not in completed_tasks:
                                slack = self._calculate_slack(graph_idx, tgt, current_time)
                                ready_tasks.append((tgt, slack))
                                ready_tasks.sort(key=lambda x: x[1])
                
                schedule.append((graph_idx, copy_idx, copy_schedule))
            
            for copy_idx in range(n_real_copies, n_copies):  # copy tasks scheduling
                start_offset = copy_idx * period
                copy_schedule = [None] * n_tasks
                parent_copy_idx = 0
                parent_schedule = [s for g, c, s in schedule if g == graph_idx and c == parent_copy_idx][0]
                for task_id in range(n_tasks):
                    parent_start, parent_end = parent_schedule[task_id]
                    duration = graph['tasks'][task_id]['duration']
                    copy_schedule[task_id] = [parent_start + (copy_idx - parent_copy_idx) * period, 
                                             parent_end + (copy_idx - parent_copy_idx) * period]
                schedule.append((graph_idx, copy_idx, copy_schedule))
            
            task_offset += n_tasks * n_real_copies
        
        print(f"Schedule: {[(g, c, s) for g, c, s in schedule]}")
        print(f"Link usage: {link_usage}")
        return schedule, link_usage

    def _evaluate(self, x, out, *args, **kwargs):
        pe_alloc = x[:self.n_pe_types]
        ic_alloc = x[self.n_pe_types:self.n_pe_types + self.n_ic_types]
        link_alloc = x[self.n_pe_types + self.n_ic_types:self.n_pe_types + self.n_ic_types + self.n_link_types]
        task_assign = x[self.n_pe_types + self.n_ic_types + self.n_link_types:self.n_pe_types + self.n_ic_types + self.n_link_types + self.n_tasks]
        core_assign = x[self.n_pe_types + self.n_ic_types + self.n_link_types + self.n_tasks:self.n_pe_types + self.n_ic_types + self.n_link_types + self.n_tasks + sum(1 for pe in self.pe_database if pe['type'] == 'core')]
        link_connect = x[self.n_pe_types + self.n_ic_types + self.n_link_types + self.n_tasks + sum(1 for pe in self.pe_database if pe['type'] == 'core'):]

        # cost calculation
        pe_cost = sum(self.pe_database[i]['price'] * pe_alloc[i] for i in range(self.n_pe_types) if pe_alloc[i] > 0)
        ic_cost = sum(self.ic_database[i]['price'] * ic_alloc[i] for i in range(self.n_ic_types) if ic_alloc[i] > 0)
        link_cost = sum(self.link_database[i]['price'] * link_alloc[i] for i in range(self.n_link_types) if link_alloc[i] > 0)
        # cost calculation base on links
        schedule, link_usage = self._schedule_tasks(task_assign, link_alloc, link_connect, core_assign)
        link_cost += link_usage * self.link_database[0]['price'] 
        cost = pe_cost + ic_cost + link_cost

        # power calculation
        pe_power = sum(self.pe_database[i]['power'] * pe_alloc[i] for i in range(self.n_pe_types) if pe_alloc[i] > 0)
        link_power = sum(self.link_database[i]['power'] * link_alloc[i] for i in range(self.n_link_types) if link_alloc[i] > 0)
        link_power += link_usage * self.link_database[0]['power']
        power = pe_power + link_power

        out["F"] = [cost, power]
        print(f"Cost breakdown: PE={pe_cost}, IC={ic_cost}, Link={link_cost}, Total={cost}")
        print(f"Power breakdown: PE={pe_power}, Link={link_power}, Total={power}")
        print(f"Link usage: {link_usage}")
