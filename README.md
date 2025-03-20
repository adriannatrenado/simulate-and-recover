
This project tests whether the EZ diffusion model can accurately recover its own parameters from simulated data. The EZ model is a simplified version of the Drift Diffusion Model (DDM), commonly used to analyze decision-making based on reaction time and accuracy. The goal is to check if the model’s parameter estimation is consistent—meaning, if we generate data using known parameters, can the model correctly estimate them back?  
  
To run this test, we first randomly selected realistic parameter values within given ranges:  
- Boundary separation (a): `0.5 ≤ a ≤ 2` (controls decision caution)  
- Drift rate (v): `0.5 ≤ v ≤ 2` (measures evidence accumulation speed)  
- Nondecision time (t): `0.1 ≤ t ≤ 0.5` (accounts for encoding/motor response delays)  

Using these parameters, we simulated reaction time (RT) data using the EZ diffusion model’s forward equations. This mimics real-world decision-making scenarios where people make speeded choices under uncertainty.  

Next, we used the inverse EZ diffusion equations to estimate the original parameters from the simulated data. This process was repeated 1,000 times for each of three different sample sizes:  
- N = 10 (small dataset)  
- N = 40 (moderate dataset)  
- N = 4000 (large dataset)  

The key idea behind this test is that, if the EZ diffusion model is valid, the estimated parameters should closely match the true ones, with bias averaging to zero and squared error decreasing as N increases.  

The results showed expected patterns:  

- With N = 10, parameter estimates were highly unstable, meaning the model couldn’t reliably recover the original values.  
- With N = 40, estimates became more stable, but there were still noticeable deviations.  
- With N = 4000, the estimated parameters were very close to the true ones, with minimal bias and low error.  

This confirms that the EZ diffusion model performs well when given enough data, but struggles with small sample sizes.  

This experiment demonstrates that the EZ diffusion model is consistent and reliable under the right conditions. When given large datasets, it accurately recovers parameters, making it a useful tool for decision-making research. However, small sample sizes lead to unreliable estimates, which is a major limitation. This highlights the importance of data quantity in cognitive modeling—more data leads to more accurate parameter recovery.  

Overall, this project provided a basic but crucial consistency check for the EZ diffusion model. While real-world data is often more complex, this test ensures that at least under controlled conditions, the model can correctly estimate parameters from its own generated data. Future work could involve testing the model under more complex scenarios or comparing it to other parameter estimation methods.  
