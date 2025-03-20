import numpy as np
import pandas as pd

BOUNDARY_SEP_RANGE = (0.5, 2.0)  # boundary separation (a)
DRIFT_RATE_RANGE = (0.5, 2.0)  # drift rate (v)
NONDECISION_TIME_RANGE = (0.1, 0.5)  # nondecision time (t)

N_SIZES = [10, 40, 4000]
ITERATIONS = 1000

def generate_true_parameters():
    """pick random values for a, v, and t within the given ranges"""
    a = np.random.uniform(*BOUNDARY_SEP_RANGE)
    v = np.random.uniform(*DRIFT_RATE_RANGE)
    t = np.random.uniform(*NONDECISION_TIME_RANGE)
    return a, v, t

def forward_equations(a, v, t):
    """Use  EZ diffusion model equations 2 calculate predicted accuracy and response times."""
    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)  # Expected accuracy
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))  # Expected mean response time
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))  # Expected variance in RT
    return R_pred, M_pred, V_pred

def simulate_observed_statistics(R_pred, M_pred, V_pred, N):
    """Simulate accuracy and reaction times based on the predicted values."""
    if N < 2:
        return None, None, None  # Need at least 2 samples for variance
    
    Robs = np.random.binomial(N, R_pred) / N  # Simulated accuracy
    Mobs = np.random.normal(M_pred, np.sqrt(V_pred / max(N, 1)))  # Simulated mean RT
    Vobs = np.random.gamma((N - 1) / 2, 2 * V_pred / max(N - 1, 1))  # Simulated RT variance
    return Robs, Mobs, Vobs

def inverse_equations(Robs, Mobs, Vobs):
    """Estimate a, v, and t using the observed data and EZ inverse equations."""
    if Robs in [0, 1] or Vobs <= 0:
        return None, None, None  # Avoid division errors
    
    eps = 1e-6  # Small value to prevent division by zero
    L = np.log(Robs / (1 - Robs))
    
    v_est = np.sign(Robs - 0.5) * np.sqrt(L * ((Robs**2 * L) - (Robs * L) + (Robs - 0.5)) / (Vobs + eps))
    a_est = L / (v_est + eps)  # Estimate boundary separation
    t_est = Mobs - (a_est / (2 * (v_est + eps))) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))  # Estimate nondecision time
    
    return a_est, v_est, t_est

def simulate_and_recover():
    """Run the simulation and parameter recovery process, then save the results to a CSV file."""
    results = []
    
    for N in N_SIZES:
        for _ in range(ITERATIONS):
            a, v, t = generate_true_parameters()  # Generate true values
            R_pred, M_pred, V_pred = forward_equations(a, v, t)  # Compute expected values
            Robs, Mobs, Vobs = simulate_observed_statistics(R_pred, M_pred, V_pred, N)  # Simulate observed data
            a_est, v_est, t_est = inverse_equations(Robs, Mobs, Vobs)  # Estimate parameters
            
            if None in (a_est, v_est, t_est):
                continue  # Skip invalid cases
            
            # Compute bias (how far off the estimates are)
            bias_a, bias_v, bias_t = a_est - a, v_est - v, t_est - t
            se_a, se_v, se_t = bias_a**2, bias_v**2, bias_t**2  # Squared error

            results.append([N, bias_a, bias_v, bias_t, se_a, se_v, se_t])
    
    # Save the results
    df = pd.DataFrame(results, columns=['N', 'Bias_a', 'Bias_v', 'Bias_t', 'SE_a', 'SE_v', 'SE_t'])
    df.to_csv("results.csv", index=False, float_format="%.6f")
    
if __name__ == "__main__":
    simulate_and_recover()
