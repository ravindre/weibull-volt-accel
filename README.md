# Weibull-Volt-Accel

**Fast, adaptive Monte Carlo framework to estimate voltage-acceleration (VAF) and Weibull shape/scale for reliability tests.**

> 90 % fewer simulations, full performance telemetry, DuckDB streaming, and blog-quality dashboards — all in one repo.

## 🌟 Highlights
- **Adaptive early-stop**: ends Monte Carlo once the 95 % CI on gamma is tight enough.
- **Penalizer sweep**: fits lifelines’ Weibull AFT with {0.001, 0.003, 0.005} penalizers and picks best by AIC/BIC/holdout.
- **DuckDB streaming**: zero memory bloat, instant SQL queries.
- **Performance dashboard**: one PNG summarizes timing, memory, CI convergence, accuracy scatter.
- **Mixed-effects ready**: architecture supports wafer/tray random intercepts.

## Quick start

