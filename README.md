# AstroPark
Astronomical Research

!!!This depository is in development!!! Please be carefult!!!!
### introduction
Here, I develop a "spot-modeling code" for Kepler data.
Kepler stars with large spots show quasi-periodic brightness variation caused by stellar rotations.
These data enable us to reconstruct the stellar surface spot distributions.

### Application to a special case
Now we are developing the code, especially for a solar-type star: Kepler-17.
Therefore, the default parameters are set to the stellar parameters of Kepler-17 (e.g., rotational period.)

### Detail Method:
1. **Bayesian estimation** of spot parameters.
  (rotational period, carrington longitude, maximum area, emerge & decay rate, and peak time)
2. using **MCMC algorithm** (metropolis algorithm).
3. proposed distributions are chosen to be **Gaussian functions** (normal distribution), 
  and the standard deviations are adaptively modified in MCMC (cf. Araki et al. 2013)
4. **Parallel tempering** algorithm (cf. Araki et al. 2013)
