"""
modeling.py
System, equations, parameters, etc.

author: Scott Renegado
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def compute_ancestral_virulence(ode_parameters):
    """Singular strategy of the host-parasite system"""
    return ode_parameters['d']*(-ode_parameters['b'] - ode_parameters['gamma_P'])/(ode_parameters['d'] - 1)


def set_default_ode_parameters():
    """Return ODE parameters with default values"""
    RESISTANCE = 0.5
    MAX_PER_CAPITA_BIRTH_RATE = 1.0
    NATURAL_MORTALITY_RATE = 0.5#0.25
    DEF_SYMBIONT_COST_STRENGTH = 0.1#0.25
    DEF_SYMBIONT_COST_SHAPE = 2.0
    POWER_LAW = 0.5
    DENSITY_DEPENDENT_COMPETITION_STRENGTH = 0.5#0.25
    ADDITIONAL_MORTALITY_RATE_DEF_SYMBIONT = 0.1
    MAX_TRANSMISSION_RATE_DEF_SYMBIONT = 5.0#1.0
    TRANSMISSION_RATE_PARASITE_SCALING_CONST = 5.0#2.0
    RECOVERY_RATE_DEF_SYMBIONT = 0.1#0.05
    RECOVERY_RATE_PARASITE = 0.1#0.05
    
    ode_parameters = {
        'alpha_P': np.nan,
        'resistance': RESISTANCE,
        'c1': DEF_SYMBIONT_COST_STRENGTH,
        'a': MAX_PER_CAPITA_BIRTH_RATE,
        'b': NATURAL_MORTALITY_RATE,
        'c2': DEF_SYMBIONT_COST_SHAPE,
        'd': POWER_LAW,
        'q': DENSITY_DEPENDENT_COMPETITION_STRENGTH, 
        'alpha_D': ADDITIONAL_MORTALITY_RATE_DEF_SYMBIONT,
        'beta_hat_D': MAX_TRANSMISSION_RATE_DEF_SYMBIONT,
        'beta_hat_P': TRANSMISSION_RATE_PARASITE_SCALING_CONST,
        'gamma_D': RECOVERY_RATE_DEF_SYMBIONT,
        'gamma_P': RECOVERY_RATE_PARASITE
    }
    ode_parameters['alpha_P'] = compute_ancestral_virulence(ode_parameters)

    return ode_parameters


def build_alpha_B():
    """Additional mortality rate of having both microbes"""
    return lambda alpha_D, alpha_P: alpha_D + alpha_P


def build_beta_P(beta_hat_P, d):
    """Transmission rate of parasite"""
    return lambda alpha_P: beta_hat_P*(alpha_P**d)


def build_beta_D(beta_hat_D, c1, c2):
    """Transmission rate of defensive symbiont"""
    cost = lambda resistance: c1*(1.0-np.exp(c2*resistance))/(1.0-np.exp(c2)) if c2 != 0.0 else c1*resistance
    return lambda resistance: beta_hat_D*(1.0-cost(resistance))


def make_system(ode_parameters):
    """
    Return right-hand side of the population dynamics.
    For input to solve_ivp's fun parameter. 
    """
    def system(
        t, y, alpha_P,
        resistance=ode_parameters['resistance'],
        c1=ode_parameters['c1'],
        a=ode_parameters['a'],
        b=ode_parameters['b'],
        c2=ode_parameters['c2'],
        d=ode_parameters['d'],
        q=ode_parameters['q'], 
        alpha_D=ode_parameters['alpha_D'],
        beta_hat_D=ode_parameters['beta_hat_D'],
        beta_hat_P=ode_parameters['beta_hat_P'],
        gamma_D=ode_parameters['gamma_D'],
        gamma_P=ode_parameters['gamma_P']
    ):
        """RHS of resident population dynamics"""

        # Model parameters
        H, D, P, B = y      # classes
        N = H + D + P + B   # total host population
        nu = N*(a-q*N)      # birth rate of new hosts    

        alpha_B = build_alpha_B()
        beta_P = build_beta_P(beta_hat_P, d)
        beta_D = build_beta_D(beta_hat_D, c1, c2)
        
        # System of ODEs
        dH = nu - (b + beta_D(resistance)*(D + B) + beta_P(alpha_P)*(P + B))*H + gamma_D*D + gamma_P*P
        dD = beta_D(resistance)*H*(D + B) - (b + gamma_D + alpha_D + beta_P(alpha_P)*(1-resistance)*(P + B))*D + gamma_P*B
        dP = beta_P(alpha_P)*H*(P + B) - (b + gamma_P + alpha_P + beta_D(resistance)*(D + B))*P + gamma_D*B
        dB = beta_D(resistance)*P*(D + B) + beta_P(alpha_P)*(1-resistance)*D*(P + B) - (b + alpha_B(alpha_D,alpha_P) + gamma_D + gamma_P)*B
        
        return dH, dD, dP, dB
    return system


def make_discretized_system(ode_parameters):
    """
    Return right-hand side of the discretized population dynamics.
    For input to solve_ivp's fun parameter. 
    """
    def discretized_system(
        t, state, present_virulence_values,
        resistance=ode_parameters['resistance'],
        c1=ode_parameters['c1'],
        a=ode_parameters['a'],
        b=ode_parameters['b'],
        c2=ode_parameters['c2'],
        d=ode_parameters['d'],
        q=ode_parameters['q'], 
        alpha_D=ode_parameters['alpha_D'],
        beta_hat_D=ode_parameters['beta_hat_D'],
        beta_hat_P=ode_parameters['beta_hat_P'],
        gamma_D=ode_parameters['gamma_D'],
        gamma_P=ode_parameters['gamma_P']
    ):
        """
        Right-hand side of discretized population dynamics.

        state is of size 

            1 (H) + 1 (D) + number_of_present_virulence_values (P) + number_of_present_virulence_values (B) 
            = 2*(number_of_present_virulence_values + 1) 

        where number_of_present_virulence_values is the number of parasite strains 
        that are present in the system, that is, whose density is above the extinction threshold.
        """
        number_of_present_virulence_values = len(present_virulence_values)

        H = state[0]
        D = state[1]
        P = state[2:number_of_present_virulence_values+2] # is of length number_of_present_virulence_values
        B = state[number_of_present_virulence_values+2:]  # is of length number_of_present_virulence_values
        N = H + D + np.sum(P) + np.sum(B)
        nu = N*(a-q*N)

        alpha_B = build_alpha_B()
        beta_P = build_beta_P(beta_hat_P, d)
        beta_D = build_beta_D(beta_hat_D, c1, c2)
        alpha_D_vector = np.full(shape=(number_of_present_virulence_values,), fill_value=alpha_D)
        
        dH = nu - ( b + beta_D(resistance)*(D + np.sum(B)) + np.sum(beta_P(present_virulence_values)*(P + B)) )*H + gamma_D*D + gamma_P*np.sum(P)
        dD = beta_D(resistance)*H*(D + np.sum(B)) - ( b + gamma_D + alpha_D + np.sum(beta_P(present_virulence_values)*(1-resistance)*(P + B)) )*D + gamma_P*np.sum(B)
        dP = beta_P(present_virulence_values)*H*(P + B) - ( b + gamma_P + present_virulence_values + beta_D(resistance)*(D + np.sum(B)) )*P + gamma_D*B
        dB = beta_D(resistance)*P*(D + np.sum(B)) + beta_P(present_virulence_values)*(1-resistance)*D*(P + B) - (b + alpha_B(alpha_D_vector,present_virulence_values) + gamma_D + gamma_P)*B

        return np.hstack((dH, dD, dP, dB))
    return discretized_system


def make_discretized_system_coevolution(ode_parameters):
    """
    Return right-hand side of the discretized population dynamics.
    For input to solve_ivp's fun parameter. 
    """
    def discretized_system_coevolution(
        t, state, present_virulence_values, present_resistance_values,
        c1=ode_parameters['c1'],
        a=ode_parameters['a'],
        b=ode_parameters['b'],
        c2=ode_parameters['c2'],
        d=ode_parameters['d'],
        q=ode_parameters['q'], 
        alpha_D=ode_parameters['alpha_D'],
        beta_hat_D=ode_parameters['beta_hat_D'],
        beta_hat_P=ode_parameters['beta_hat_P'],
        gamma_D=ode_parameters['gamma_D'],
        gamma_P=ode_parameters['gamma_P']
    ):
        """
        Right-hand side of discretized population dynamics.

        state is of size 

            1 (H) + number_of_present_resistance_values (D) + number_of_present_virulence_values (P) 
            + number_of_present_resistance_values*number_of_present_virulence_values (B) 
            = (number_of_resistaance_values + 1)*(number_of_present_virulence_values + 1) 

        where number_of_present_resistance_values and number_of_present_virulence_values is the number of respective microbe strains 
        that are present in the system, that is, whose density is above the extinction threshold.

        B is a matrix of size number_of_present_resistance_values x number_of_present_virulence_values.
        The ij-th entry of B is the density of those harbouring present defensive symbiont i and infected with present parasite j.  
        """
        number_of_present_resistance_values = len(present_resistance_values)
        number_of_present_virulence_values = len(present_virulence_values)

        H = state[0]
        D = state[1:number_of_present_resistance_values+1]
        P = state[number_of_present_resistance_values+1:number_of_present_virulence_values+number_of_present_resistance_values+1] 
        B = state[number_of_present_virulence_values+number_of_present_resistance_values+1:]
        B = np.reshape(B, (number_of_present_resistance_values, number_of_present_virulence_values)) 

        sum_D = np.sum(D); sum_P = np.sum(P); sum_B = np.sum(B)

        N = H + sum_D + sum_P + sum_B
        nu = N*(a-q*N)

        alpha_B = build_alpha_B()
        beta_P = build_beta_P(beta_hat_P, d)
        beta_D = build_beta_D(beta_hat_D, c1, c2)
        alpha_D_vector = np.full(shape=(number_of_present_virulence_values,), fill_value=alpha_D)
        alpha_B_matrix = np.tile(alpha_B(alpha_D_vector,present_virulence_values), (number_of_present_resistance_values,1))

        rows = 0; columns = 1
        
        sum_B_rows = np.sum(B, rows)
        sum_B_columns = np.sum(B, columns)

        dH = nu - ( b + np.sum(beta_D(present_resistance_values)*(D + sum_B_columns)) + np.sum(beta_P(present_virulence_values)*(P + sum_B_rows)) )*H + gamma_D*sum_D + gamma_P*sum_P
        dD = beta_D(present_resistance_values)*H*(D + sum_B_columns) - ( b + gamma_D + alpha_D + (1-present_resistance_values)*np.sum(beta_P(present_virulence_values)*(P + sum_B_rows)) )*D + gamma_P*sum_B_columns
        dP = beta_P(present_virulence_values)*H*(P + sum_B_rows) - ( b + gamma_P + present_virulence_values + np.sum(beta_D(present_resistance_values)*(D + sum_B_columns)) )*P + gamma_D*sum_B_rows
        dB = np.reshape(beta_D(present_resistance_values)*(D + sum_B_columns), (number_of_present_resistance_values, 1))*P \
            + beta_P(present_virulence_values)*(P + sum_B_rows)*np.reshape(D*(1 - present_resistance_values), (number_of_present_resistance_values,1)) \
            - (b + alpha_B_matrix + gamma_D + gamma_P)*B
        dB = np.reshape(dB, (number_of_present_resistance_values*number_of_present_virulence_values,))

        return np.hstack((dH, dD, dP, dB))
    return discretized_system_coevolution

def rk45updateParasite(ode_parameters, system, present_virulence_values, t, state, dt, tol = 1e-2):
    """
    RK45 update using system and ode_parameters over a time-step dt.
    Code will calculate new time steps adaptively.
    """

    # RK scheme
    A = [0, 2/9, 1/3, 3/4, 1, 5/6]
    B = [[], [2/9], [1/12, 1/4], [69/128, -243/128, 135/64], [-17/12, 27/4, -27/5, 16/15], [65/432, -5/16, 13/16, 4/27, 5/144]]
    C = [1/9, 0, 9/20, 16/45, 1/12]
    CH = [47/450, 0, 12/25, 32/225, 1/30, 6/25]
    CT = [1/150, 0, -3/100, 16/75, 1/20, -6/25]

    # We haven't yet done the step
    doneStep = False

    # While loop
    while not doneStep:

        # RK45 k values
        k1 = dt*system(t + A[0]*dt, state, present_virulence_values)
        k2 = dt*system(t + A[1]*dt, state + B[1][0]*k1, present_virulence_values)
        k3 = dt*system(t + A[2]*dt, state + B[2][0]*k1 + B[2][1]*k2, present_virulence_values)
        k4 = dt*system(t + A[3]*dt, state + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3, present_virulence_values)
        k5 = dt*system(t + A[4]*dt, state + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 + B[4][3]*k4, present_virulence_values)
        k6 = dt*system(t + A[5]*dt, state + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 + B[5][3]*k4 + B[5][4]*k5, present_virulence_values)

        # Calculate truncation error
        TE = np.linalg.norm(CT[0]*k1 + CT[1]*k2 + CT[2]*k3 + CT[3]*k4 + CT[4]*k5 + CT[5]*k6)

        # New time-step
        dtNew = 0.9*dt*(tol/TE)**0.2

        # Check if we accept
        if TE <= tol:
            doneStep = True
        else:
            dt = dtNew

    # print(f"Timestep = {dtNew}")

    # Return the new timestep and state
    return(dtNew, state + CH[0]*k1 + CH[1]*k2 + CH[2]*k3 + CH[3]*k4 + CH[4]*k5 + CH[5]*k6)

def rk4update(ode_parameters, system, t, state, present_virulence_values, present_resistance_values, dt):
    """RK4 update step using system and ode_parameters over a time step dt"""

    # RK4 k values
    k1 = system(t, state, present_virulence_values, present_resistance_values)
    k2 = system(t + dt/2, state + dt*k1/2, present_virulence_values, present_resistance_values)
    k3 = system(t + dt/2, state + dt*k2/2, present_virulence_values, present_resistance_values)
    k4 = system(t + dt, state + dt*k3, present_virulence_values, present_resistance_values)

    # Return updated state
    return(state + (k1 + 2*k2 + 2*k3 + k4)*dt/6)

def rk45update(ode_parameters, system, t, state, present_virulence_values, present_resistance_values, dt, tol = 1e-2):
    """
    RK45 update using system and ode_parameters over a time-step dt.
    Code will calculate new time steps adaptively.
    """

    # RK scheme
    A = [0, 2/9, 1/3, 3/4, 1, 5/6]
    B = [[], [2/9], [1/12, 1/4], [69/128, -243/128, 135/64], [-17/12, 27/4, -27/5, 16/15], [65/432, -5/16, 13/16, 4/27, 5/144]]
    C = [1/9, 0, 9/20, 16/45, 1/12]
    CH = [47/450, 0, 12/25, 32/225, 1/30, 6/25]
    CT = [1/150, 0, -3/100, 16/75, 1/20, -6/25]

    # We haven't yet done the step
    doneStep = False

    # While loop
    while not doneStep:

        # RK45 k values
        k1 = dt*system(t + A[0]*dt, state, present_virulence_values, present_resistance_values)
        k2 = dt*system(t + A[1]*dt, state + B[1][0]*k1, present_virulence_values, present_resistance_values)
        k3 = dt*system(t + A[2]*dt, state + B[2][0]*k1 + B[2][1]*k2, present_virulence_values, present_resistance_values)
        k4 = dt*system(t + A[3]*dt, state + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3, present_virulence_values, present_resistance_values)
        k5 = dt*system(t + A[4]*dt, state + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 + B[4][3]*k4, present_virulence_values, present_resistance_values)
        k6 = dt*system(t + A[5]*dt, state + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 + B[5][3]*k4 + B[5][4]*k5, present_virulence_values, present_resistance_values)

        # Calculate truncation error
        TE = np.linalg.norm(CT[0]*k1 + CT[1]*k2 + CT[2]*k3 + CT[3]*k4 + CT[4]*k5 + CT[5]*k6)

        # New time-step
        dtNew = 0.9*dt*(tol/TE)**0.2

        # Check if we accept
        if TE <= tol:
            doneStep = True
        else:
            dt = dtNew

    # print(f"Timestep = {dtNew}")

    # Return the new timestep and state
    return(dtNew, state + CH[0]*k1 + CH[1]*k2 + CH[2]*k3 + CH[3]*k4 + CH[4]*k5 + CH[5]*k6)

def run_ode_solver(ode_parameters, system):
    """Given parameters, run scipy's solve_ivp function and output times series Ht, Dt, Pt, Bt"""
    initial_state = (8, 1, 1, 0)
    t_span = (0, 100)
    times = np.linspace(t_span[0], t_span[1], 100)
    
    soln = solve_ivp(system, t_span, initial_state, t_eval=times, args=(
        ode_parameters['alpha_P'],
        ode_parameters['resistance'],
        ode_parameters['c1'],
        ode_parameters['a'],
        ode_parameters['b'],
        ode_parameters['c2'],
        ode_parameters['d'],
        ode_parameters['q'],
        ode_parameters['alpha_D'],
        ode_parameters['beta_hat_D'],
        ode_parameters['beta_hat_P'],
        ode_parameters['gamma_D'],
        ode_parameters['gamma_P']
    ))
    Ht, Dt, Pt, Bt = soln.y

    return Ht, Dt, Pt, Bt


def run_ode_solver_on_discretized_system(ode_parameters, system, initial_state, present_virulence_values):
    """
    Run scipy's solve_ivp function and output discrete times series Ht, Dt, Pt, Bt.
    initial_state is a tuple of size 2*(len(present_virulence_values)+1)
    """
    t_span = (0, 100)
    times = np.linspace(t_span[0], t_span[1], 100)
    dt = 1e-1

    # soln = solve_ivp(system, t_span, initial_state, t_eval=times, args=(
    #     present_virulence_values,
    #     ode_parameters['resistance'],
    #     ode_parameters['c1'],
    #     ode_parameters['a'],
    #     ode_parameters['b'],
    #     ode_parameters['c2'],
    #     ode_parameters['d'],
    #     ode_parameters['q'],
    #     ode_parameters['alpha_D'],
    #     ode_parameters['beta_hat_D'],
    #     ode_parameters['beta_hat_P'],
    #     ode_parameters['gamma_D'],
    #     ode_parameters['gamma_P'],
    # ))
    # discrete_time_series = soln.y

    discrete_time_series = []
    state = initial_state
    discrete_time_series.append(state)
    tInd = 0
    t = 0
    while t <= t_span[1]:
        dt, state = rk45updateParasite(ode_parameters, system, present_virulence_values, t, state, dt)
        t += dt
        discrete_time_series.append(state)
    discrete_time_series = np.array(discrete_time_series).transpose()

    return discrete_time_series


def run_ode_solver_on_discretized_system_coevolution(ode_parameters, system, initial_state, present_virulence_values, present_resistance_values):
    """
    Run scipy's solve_ivp function and output discrete times series Ht, Dt, Pt, Bt.
    initial_state is a tuple of size (len(present_resistance_values)+1)*(len(present_virulence_values)+1)
    """
    t_span = (0, 100)
    times = np.linspace(t_span[0], t_span[1], 1001)
    dt = 1e-1
    
    # soln = solve_ivp(system, t_span, initial_state, t_eval=times, args=(
    #     present_virulence_values,
    #     present_resistance_values,
    #     ode_parameters['c1'],
    #     ode_parameters['a'],
    #     ode_parameters['b'],
    #     ode_parameters['c2'],
    #     ode_parameters['d'],
    #     ode_parameters['q'],
    #     ode_parameters['alpha_D'],
    #     ode_parameters['beta_hat_D'],
    #     ode_parameters['beta_hat_P'],
    #     ode_parameters['gamma_D'],
    #     ode_parameters['gamma_P'],
    # ))
    # discrete_time_series = soln.y

    discrete_time_series = []
    state = initial_state
    discrete_time_series.append(state)
    tInd = 0
    t = 0
    while t <= t_span[1]:
        dt, state = rk45update(ode_parameters, system, t, state, present_virulence_values, present_resistance_values, dt)
        t += dt
        discrete_time_series.append(state)
    discrete_time_series = np.array(discrete_time_series).transpose()

    return discrete_time_series

def approximate_steady_state(Ht, Dt, Pt, Bt):
    """Take final population densities as an approximation for the steady state"""
    steady_state = {'H': Ht[Ht.size-1], 'D': Dt[Dt.size-1], 'P': Pt[Pt.size-1], 'B': Bt[Bt.size-1]}
    return steady_state


def build_initial_state_coevolution(initial_resistance_index, initial_virulence_index, resistance_vector, virulence_vector):
    """Create initial state for discretized system (coevolution)"""
    NUMBER_OF_RESISTANCE_VALUES = len(resistance_vector)
    NUMBER_OF_VIRULENCE_VALUES = len(virulence_vector)

    H = np.array([8])
    D = np.zeros(NUMBER_OF_RESISTANCE_VALUES); D[initial_resistance_index] = 1
    P = np.zeros(NUMBER_OF_VIRULENCE_VALUES); P[initial_virulence_index] = 1
    B = np.zeros(NUMBER_OF_VIRULENCE_VALUES*NUMBER_OF_RESISTANCE_VALUES)
    initial_state = np.hstack((H, D, P, B))

    return initial_state


def build_A(ode_parameters, steady_state_approximation):
    """
    Function A(alpha_Pm, alpha_P) of mutant trait alpha_Pm and resident trait alpha_P.
    Substitutes into invasion fitness and fitness gradient.
    """
    Hstar = steady_state_approximation['H']
    Dstar = steady_state_approximation['D']
    Bstar = steady_state_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    gamma_D = ode_parameters['gamma_D']
    gamma_P = ode_parameters['gamma_P']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: Dstar*(gamma_D*(1-y)+(1-y)*(Bstar+Dstar)*beta_D(y)+(1-y)*(alpha_Pm+b+gamma_P))\
        +Hstar*(b+gamma_D+gamma_P+(Bstar+Dstar)*beta_D(y)+alpha_B(alpha_D,alpha_Pm))


def build_B(ode_parameters, steady_state_approximation):
    """
    Function B(alpha_Pm, alpha_P) of mutant trait alpha_Pm and resident trait alpha_P
    Substitutes into invasion fitness and fitness gradient.
    """
    Dstar = steady_state_approximation['D']
    Bstar = steady_state_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    gamma_D = ode_parameters['gamma_D']
    gamma_P = ode_parameters['gamma_P']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: -gamma_D*(Bstar + Dstar)*beta_D(y) \
        + (alpha_Pm + b + gamma_P + (Bstar + Dstar)*beta_D(y))*(b + gamma_D + gamma_P + alpha_B(alpha_D, alpha_Pm))


def build_invasion_fitness(ode_parameters, steady_state_approximation):
    """
    Invasion fitness w_P(alpha_Pm, alpha_P).
    The output has only one argument (alpha_Pm) because the steady states - functions of alpha_P - are approximated.
    """
    A = build_A(ode_parameters, steady_state_approximation)   
    B = build_B(ode_parameters, steady_state_approximation)
    beta_P = build_beta_P(ode_parameters['beta_hat_P'], ode_parameters['d'])  

    return lambda alpha_Pm: A(alpha_Pm)*beta_P(alpha_Pm)/B(alpha_Pm) - 1


def build_dAdm(ode_parameters, steady_state_approximation):
    """First derivative of substition A with respect to the mutant trait"""
    Hstar = steady_state_approximation['H']
    Dstar = steady_state_approximation['D']
    y = ode_parameters['resistance']
    
    return Dstar*(1 - y) + Hstar 


def build_dBdm(ode_parameters, steady_state_approximation):
    """First derivative of substition B with respect to the mutant trait"""
    Dstar = steady_state_approximation['D']
    Bstar = steady_state_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    gamma_D = ode_parameters['gamma_D']
    gamma_P = ode_parameters['gamma_P']
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: alpha_Pm + 2*b + gamma_D + 2*gamma_P + (Bstar + Dstar)*beta_D(y) + alpha_B(alpha_D, alpha_Pm)


def build_dbetaP(beta_hat_P, d):
    """First derivative of parasite transmission rate"""
    return lambda alpha_P: d*beta_hat_P*(alpha_P**(d-1))


def build_fitness_gradient(ode_parameters, steady_state_approximation):
    """
    Fitness gradient F_P(alpha_P).
    Derivative of invasion fitness with respect to mutant trait evaluated at mutant = resident.
    """
    A = build_A(ode_parameters, steady_state_approximation)   
    B = build_B(ode_parameters, steady_state_approximation)
    beta_P = build_beta_P(ode_parameters['beta_hat_P'], ode_parameters['d'])
    dAdm = build_dAdm(ode_parameters, steady_state_approximation)
    dBdm = build_dBdm(ode_parameters, steady_state_approximation)
    dbeta_P = build_dbetaP(ode_parameters['beta_hat_P'], ode_parameters['d'])
    
    return lambda alpha_P: A(alpha_P)*dbeta_P(alpha_P)/B(alpha_P) \
        - A(alpha_P)*beta_P(alpha_P)*dBdm(alpha_P)/(B(alpha_P)**2) \
        + beta_P(alpha_P)*dAdm/B(alpha_P)


def build_d2betaP(beta_hat_P, d):
    """Second derivative of parasite transmission rate"""
    return lambda alpha_P: d*(d-1)*beta_hat_P*(alpha_P**(d-2))


def build_invasion_fitness_2nd_derivative_wrt_mutant(ode_parameters, steady_state_approximation):
    """Second derivative of invasion fitness with respect to mutant trait"""
    A = build_A(ode_parameters, steady_state_approximation)   
    B = build_B(ode_parameters, steady_state_approximation)
    beta_P = build_beta_P(ode_parameters['beta_hat_P'], ode_parameters['d'])
    dAdm = build_dAdm(ode_parameters, steady_state_approximation)
    dBdm = build_dBdm(ode_parameters, steady_state_approximation)
    dbeta_P = build_dbetaP(ode_parameters['beta_hat_P'], ode_parameters['d'])
    d2beta_P = build_d2betaP(ode_parameters['beta_hat_P'], ode_parameters['d'])

    return lambda alpha_Pm: ((A(alpha_Pm)*d2beta_P(alpha_Pm) + 2*dAdm*dbeta_P(alpha_Pm))*(B(alpha_Pm)**2) \
        - (2*A(alpha_Pm)*beta_P(alpha_Pm) + 2*A(alpha_Pm)*dBdm(alpha_Pm)*dbeta_P(alpha_Pm) + 2*beta_P(alpha_Pm)*dAdm*dBdm(alpha_Pm))*B(alpha_Pm) \
        + 2*A(alpha_Pm)*beta_P(alpha_Pm)*(dBdm(alpha_Pm)**2))/(B(alpha_Pm)**3)


def build_dAdr(ode_parameters, steady_state_approximation, steady_state_derivative_approximation):
    """First derivative of substitution A with respect to the resident trait"""
    Hstar = steady_state_approximation['H']
    Dstar = steady_state_approximation['D']
    Bstar = steady_state_approximation['B']
    dHstar = steady_state_derivative_approximation['H']
    dDstar = steady_state_derivative_approximation['D']
    dBstar = steady_state_derivative_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    gamma_D = ode_parameters['gamma_D']
    gamma_P = ode_parameters['gamma_P']
    b = ode_parameters['b']
    y = ode_parameters['resistance']


    return lambda alpha_Pm: (1 - y)*((dBstar + dDstar)*Dstar*beta_D(y) \
        + (alpha_Pm + b + gamma_D + gamma_P + (Bstar + Dstar)*beta_D(y))*dDstar) \
        + (dBstar + dDstar)*Hstar*beta_D(y) + (b + gamma_D + gamma_P + (Bstar + Dstar)*beta_D(y) + alpha_B(alpha_D,alpha_Pm))*dHstar


def build_dBdr(ode_parameters, steady_state_derivative_approximation):
    """First derivative of substitution B with respect to the resident trait"""
    dDstar = steady_state_derivative_approximation['D']
    dBstar = steady_state_derivative_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    gamma_P = ode_parameters['gamma_P']
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: (dBstar + dDstar)*(b + gamma_P + alpha_B(alpha_D,alpha_Pm))*beta_D(y)


def build_d2Adr2(ode_parameters, steady_state_approximation, steady_state_derivative_approximation, steady_state_2nd_derivative_approximation):
    """Second derivative of substitution A with respect to the resident trait"""
    Hstar = steady_state_approximation['H']
    Dstar = steady_state_approximation['D']
    Bstar = steady_state_approximation['B']
    dHstar = steady_state_derivative_approximation['H']
    dDstar = steady_state_derivative_approximation['D']
    dBstar = steady_state_derivative_approximation['B']
    d2Hstar = steady_state_2nd_derivative_approximation['H']
    d2Dstar = steady_state_2nd_derivative_approximation['D']
    d2Bstar = steady_state_2nd_derivative_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    gamma_D = ode_parameters['gamma_D']
    gamma_P = ode_parameters['gamma_P']
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: (1 - y)*(2*(dBstar + dDstar)*beta_D(y)*dDstar + (d2Bstar + d2Dstar)*Dstar*beta_D(y) \
        + (alpha_Pm + b + gamma_D + gamma_P + (Bstar + Dstar)*beta_D(y))*d2Dstar) \
        + 2*(dBstar + dDstar)*beta_D(y)*dHstar + (d2Bstar + d2Dstar)*Hstar*beta_D(y) \
        + (b + gamma_D + gamma_P + (Bstar + Dstar)*beta_D(y) + alpha_B(alpha_D,alpha_Pm))*d2Hstar


def build_d2Bdr2(ode_parameters, steady_state_2nd_derivative_approximation):
    """Second derivative of substitution B with respect to the resident trait"""
    d2Dstar = steady_state_2nd_derivative_approximation['D']
    d2Bstar = steady_state_2nd_derivative_approximation['B']
    alpha_B = build_alpha_B()
    alpha_D = ode_parameters['alpha_D']
    beta_D = build_beta_D(ode_parameters['beta_hat_D'], ode_parameters['c1'], ode_parameters['c2'])
    gamma_P = ode_parameters['gamma_P']
    b = ode_parameters['b']
    y = ode_parameters['resistance']

    return lambda alpha_Pm: (d2Bstar + d2Dstar)*(b + gamma_P + alpha_B(alpha_D,alpha_Pm))*beta_D(y)


def build_invasion_fitness_2nd_derivative_wrt_resident(
    ode_parameters, 
    steady_state_approximation, 
    steady_state_derivative_approximation, 
    steady_state_2nd_derivative_approximation
):
    A = build_A(ode_parameters, steady_state_approximation)   
    B = build_B(ode_parameters, steady_state_approximation)
    dAdr = build_dAdr(ode_parameters, steady_state_approximation, steady_state_derivative_approximation)
    dBdr = build_dBdr(ode_parameters, steady_state_derivative_approximation)
    d2Adr2 = build_d2Adr2(ode_parameters, steady_state_approximation, 
                          steady_state_derivative_approximation, steady_state_2nd_derivative_approximation)
    d2Bdr2 = build_d2Bdr2(ode_parameters, steady_state_2nd_derivative_approximation)
    beta_P = build_beta_P(ode_parameters['beta_hat_P'], ode_parameters['d'])

    return lambda alpha_Pm: (d2Adr2(alpha_Pm)*(B(alpha_Pm)**2) \
        - (A(alpha_Pm)*d2Bdr2(alpha_Pm) + 2*dAdr(alpha_Pm)*dBdr(alpha_Pm))*B(alpha_Pm) \
        + 2*A(alpha_Pm)*(dBdr(alpha_Pm)**2))*beta_P(alpha_Pm)/(B(alpha_Pm)**3)


def compute_ancestral_steady_state(ode_parameters):
    """Endemic steady state (H^*, P^*) for the host-parasite system"""
    alpha_P = compute_ancestral_virulence(ode_parameters)
    gamma_P = ode_parameters['gamma_P']
    beta_P = build_beta_P(ode_parameters['beta_hat_P'], ode_parameters['d'])
    a = ode_parameters['a']
    b = ode_parameters['b']
    q = ode_parameters['q']

    Hstar = (alpha_P + b + gamma_P)/beta_P(alpha_P)
    Pstar = (-2*Hstar*q + a - alpha_P - b + np.sqrt(4*Hstar*alpha_P*q + a**2 - 2*a*alpha_P - 2*a*b + alpha_P**2 + 2*alpha_P*b + b**2))/(2*q)

    return Hstar, Pstar


def test():
    print("TEST: make_discretized_system() and run_ode_solver_on_discretized_system()")
    ode_parameters = set_default_ode_parameters()

    MIN_VIRULENCE = 0.1
    MAX_VIRULENCE = 2.0
    NUMBER_OF_VIRULENCE_VALUES = 50
    virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)
    print("Virulence (alpha_P) vector:\n", virulence_vector)

    ancestral_virulence = compute_ancestral_virulence(ode_parameters)
    distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
    initial_virulence_index = round( (ancestral_virulence - MIN_VIRULENCE)/distance_between_virulence_values )

    present_virulence_value_indices = np.array([initial_virulence_index])
    present_virulence_values = virulence_vector[present_virulence_value_indices]
    print("Present virulence values:", present_virulence_values)

    present_resistance_values = np.array([0.5])
    print("Present resistance values:", present_resistance_values)

    print("Calling make_discretized_system() with default parameters...")
    discrete_system = make_discretized_system_coevolution(ode_parameters)

    print("Creating initial state...")
    H = np.array([8])
    D = np.array([1])
    P = np.zeros(NUMBER_OF_VIRULENCE_VALUES); P[present_virulence_value_indices] = 1
    B = np.zeros(NUMBER_OF_VIRULENCE_VALUES); B[present_virulence_value_indices] = 0
    initial_state = np.hstack((H, D, P, B))
    print("Initial state:\n", initial_state)
    
    print("Filtering initial state...")
    P = P[present_virulence_value_indices]
    B = B[present_virulence_value_indices]
    initial_state = np.hstack((H, D, P, B))
    print("Initial state filtered for initialized virulence values:\n", initial_state)

    print("Running ODE solver on discretized system...")
    discrete_time_series = run_ode_solver_on_discretized_system_coevolution(ode_parameters, discrete_system, initial_state, present_virulence_values, present_resistance_values)
    discrete_Ht, discrete_Dt, discrete_Pt, discrete_Bt = discrete_time_series
    
    print("Approximating steady state...")
    steady_state_approximation = discrete_time_series[:, -1]
    print("Steady state approximation: ", steady_state_approximation)

    print("Running ODE solver on continuous system...")
    system = make_system(ode_parameters)
    Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters, system)

    print("Plotting time series...")
    times = np.linspace(0, 100, 100)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
    ax1.plot(times, discrete_Ht, label='H')
    ax1.plot(times, discrete_Dt, label='D')
    ax1.plot(times, discrete_Pt, label='P')
    ax1.plot(times, discrete_Bt, label='B')
    ax1.set(title='discrete')
    ax2.plot(times, Ht)
    ax2.plot(times, Dt)
    ax2.plot(times, Pt)
    ax2.plot(times, Bt)
    ax2.set(title='continuous')
    ax2.set(xlabel='time')
    ax1.set(ylabel='host density')
    ax2.set(ylabel='host density')
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    test()