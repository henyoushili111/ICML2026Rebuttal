This hybrid is not a simple sequential combination of DE and PSO.  
Instead, the belief space in the cultural algorithm monitors progress and diversity signals, and adaptively decides whether the current generation should be dominated by PSO-based refinement, DE-based escape, or a mixed transition update.

## Algorithm: CA-Guided Adaptive Coupling of DE and PSO in ADEPSO

```text
Input:
    Objective F(z) = sum_i |f_i(z)|, where z = (lambda, x)
    Population size N
    Maximum iterations T
    Tolerance epsilon
    Initial DE parameters F0, CR0
    Initial PSO parameters omega0, c1, c2
    Stagnation window w
    Progress threshold tau_prog
    Diversity threshold tau_div

Initialize:
    Population P = {z_i^0}_{i=1}^N in the feasible search domain
    Velocity V = {v_i^0}_{i=1}^N
    Personal best pbest_i <- z_i^0
    Global best gbest <- argmin_{z_i^0} F(z_i^0)
    Memory bank M <- empty
    Belief space B <- InitializeBeliefSpace(P)

For t = 1, 2, ..., T do

    Step 1: Evaluate and update population state
        Evaluate F(z_i^t) for all z_i^t in P
        Update pbest_i and gbest
        Update memory bank M with elite solutions
        Update belief space B using P, gbest, and elites

    Step 2: Compute state signals
        Progress:
            rho_t = (F_best(t-w) - F_best(t)) / max(F_best(t-w), 1e-12)
        Diversity:
            delta_t = (1/N) * sum_i ||z_i^t - gbest||_2
        Fitness variance:
            sigma_t^2 = Var({F(z_i^t)})

    Step 3: Determine dominant operator
        If F(gbest) <= epsilon:
            return gbest

        If rho_t >= tau_prog:
            mode <- PSO-dominant refinement
        Else if rho_t < tau_prog and delta_t < tau_div and F(gbest) > epsilon:
            mode <- DE-triggered escape
        Else:
            mode <- mixed transition

    Step 4: CA-guided update
        If mode == PSO-dominant refinement:
            For each particle i:
                Update velocity:
                    v_i^(t+1) = omega_t * v_i^t
                                + c1 * r1 * (pbest_i - z_i^t)
                                + c2 * r2 * (gbest - z_i^t)
                Update position:
                    z_i^(t+1) = z_i^t + v_i^(t+1)

        Else if mode == DE-triggered escape:
            Adapt DE parameters F_t, CR_t
            For each individual i:
                Sample distinct r1, r2, r3
                Generate donor:
                    nu_i^(t+1) = (Lambda_t - 1) * z_r1^t
                               + (Lambda_t - 1) * gbest
                               + F_t * (z_r2^t - z_r3^t)
                Perform crossover with rate CR_t to obtain u_i^(t+1)
                Selection:
                    z_i^(t+1) = u_i^(t+1) if F(u_i^(t+1)) < F(z_i^t)
                                 else z_i^t

        Else:
            Split population into two subsets:
                top kappa * N elites -> PSO update
                remaining (1 - kappa) * N individuals -> DE update
            Merge updated individuals into P

    Step 5: Belief-space influence and population control
        Apply normative-knowledge correction from B
        If t mod 10 == 0 and |P| > 20:
            Retain top 80% individuals according to fitness
            Shrink population size accordingly

End For

Return gbest
```
