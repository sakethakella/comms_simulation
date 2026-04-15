import numpy as np
import random

Q = np.array([
    [-2.5,  2.0,  0.5], 
    [ 1.0, -3.0,  2.0],  
    [ 0.0,  1.5, -1.5]
])

def simulate_ctmc(Q, start_state, max_time):
    current_state = start_state
    current_time = 0
    history = [(current_time, current_state)]
    
    while current_time < max_time:
        exit_rate = -Q[current_state, current_state]
        if exit_rate == 0: 
            break
        dt = random.expovariate(exit_rate)
        current_time += dt
        row = Q[current_state].copy()
        row[current_state] = 0 
        next_state = random.choices(
            population=range(len(Q)),
            weights=row,
            k=1
        )[0]
        current_state = next_state
        history.append((current_time, current_state))
    return history

