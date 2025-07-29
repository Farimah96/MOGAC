pe_database = [
    {'type': 'core', 'price': 50, 'power': 5},  # PE0
    {'type': 'core', 'price': 90, 'power': 4},  # PE1
    {'type': 'core', 'price': 50, 'power': 5}   # PE2
]

ic_database = [
    {'price': 200, 'power': 0}  # IC0
]

link_database = [
    {'price': 20, 'power': 2}  # Link type 0
]

task_graphs = [
    {
        'period': 12,
        'tasks': [
            {'duration': 5, 'deadline': float('inf')},  # Task 0
            {'duration': 6, 'deadline': float('inf')},  # Task 1
            {'duration': 7, 'deadline': float('inf')}   # Task 2
        ],
        'edges': [
            (0, 2, 2),  # 2 packets
            (1, 2, 3)   # 3 packets
        ]
    },
    {
        'period': 13,
        'tasks': [
            {'duration': 4, 'deadline': float('inf')},  # Task 0
            {'duration': 5, 'deadline': float('inf')}   # Task 1
        ],
        'edges': [
            (0, 1, 2)  # 2 packets
        ]
    }
]