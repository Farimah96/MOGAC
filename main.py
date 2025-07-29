from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.cluster import KMeans
import numpy as np
from problem import Problem
from sampling import MOGACSampling
from crossover import MOGACCrossover
from mutation import MOGACMutation
from data import pe_database, ic_database, link_database, task_graphs



problem = Problem(
    n_pe_types=3,
    n_ic_types=1,
    n_link_types=1,
    task_graphs=task_graphs,
    pe_database=pe_database,
    ic_database=ic_database,
    link_database=link_database,
    real_copy_ratio=0.2
)

algorithm = NSGA2(
    pop_size=100,
    sampling=MOGACSampling(problem),
    crossover=MOGACCrossover(prob=0.9),
    mutation=MOGACMutation(prob=0.1),
    eliminate_duplicates=True,
    n_offsprings=50  # Simulate cluster reproduction
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=1,
    verbose=True
)

if res.F is not None:
    kmeans = KMeans(n_clusters=3, random_state=1)
    clusters = kmeans.fit_predict(res.F)
    print("Clustering results:")
    for i, cluster in enumerate(clusters):
        print(f"Solution {i}: Cost={res.F[i][0]}, Power={res.F[i][1]}, Cluster={cluster}")
        
        
# Add canvas panel to display the chart
if res.F is not None:
    import matplotlib.pyplot as plt

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(res.F[:, 0], res.F[:, 1], c='blue', label='Solutions')
    plt.xlabel('Cost')
    plt.ylabel('Power')
    plt.title('Pareto Front: Cost vs Power')
    plt.legend()
    plt.grid(True)

    # Display the chart
    plt.show()

    # Perform clustering and print results
    kmeans = KMeans(n_clusters=3, random_state=1)
    clusters = kmeans.fit_predict(res.F)
    print("Clustering results:")
    for i, cluster in enumerate(clusters):
        print(f"Solution {i}: Cost={res.F[i][0]}, Power={res.F[i][1]}, Cluster={cluster}")