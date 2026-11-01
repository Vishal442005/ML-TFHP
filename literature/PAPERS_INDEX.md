# Benchmark Literature Index: Fractional Hawkes Processes & Rough Volatility

This directory contains full-text PDFs of the seminal and latest benchmark papers
in the exact mathematical and empirical domain of the Mittag-Leffler Tempered
Fractional Hawkes Process (ML-TFHP).

| Year | Authors | Title | Venue | arXiv / Local PDF | Key Relevance to ML-TFHP |
|:---:|---|---|---|---|---|
| 2024 | Neha Gupta, Aditya Maheshwari | **Hawkes Process with Tempered Mittag-Leffler Kernel** | *arXiv Preprint* | [2408.11695](file:///E:/Financial/literature/2024_2408.11695_Hawkes_Process_Tempered_Mittag_Leffler.pdf) | Examines theoretical Laplace transforms of univariate Hawkes with tempered Mittag-Leffler kernels; lacks fast $\mathcal{O}(NM)$ state-space estimation, multivariate cross-excitation, and empirical microstructure backtesting. |
| 2024 | Neha Gupta, Aditya Maheshwari | **Tempered Fractional Hawkes Process and Its Generalization** | *arXiv Preprint* | [2405.09966](file:///E:/Financial/literature/2024_2405.09966_Tempered_Fractional_Hawkes_Process.pdf) | Studies Bochner subordination of Hawkes processes via inverse tempered stable subordinators. Purely theoretical; operational inference in observable trading clocks is intractable. |
| 2023 | Cassien Habyarimana, Jane Aduda, Enrico Scalas, Jing Chen, Alan G. Hawkes, Federico Polito | **A Fractional Hawkes Process II: Further Characterization of the Process** | *Physica A: Statistical Mechanics and its Applications* | [2211.02583](file:///E:/Financial/literature/2022_2211.02583_Fractional_Hawkes_Process_II.pdf) | Co-authored with Alan G. Hawkes. Characterizes untempered fractional Mittag-Leffler intensity, confirming critical non-integrability and proving need for our exponential tempering. |
| 2020 | Antoine Fosset, Jean-Philippe Bouchaud, Michael Benzaquen | **Non-parametric Estimation of Quadratic Hawkes Processes for Order Book Events** | *Quantitative Finance* | [2005.05730](file:///E:/Financial/literature/2020_2005.05730_Quadratic_Hawkes_Process_Volatility.pdf) | Formulates non-parametric Wiener-Hopf calibration of order book reflexivity and volatility feedback. Inspires our continuous resolvent operator derivation. |
| 2019 | Eduardo Abi Jaber, Martin Larsson, Sergio Pulido | **Affine Volterra Processes** | *Annals of Applied Probability* | [1708.08796](file:///E:/Financial/literature/2019_1708.08796_Affine_Volterra_Processes.pdf) | Establishes the axiomatic foundation of completely monotone kernels, Bernstein's theorem, and infinite-dimensional Markovian lifting. Provides theoretical basis for our quadrature representation. |
| 2019 | Omar El Euch, Mathieu Rosenbaum | **The Characteristic Function of Rough Heston Models** | *Mathematical Finance* | [1609.02108](file:///E:/Financial/literature/2019_1609.02108_The_Characteristic_Function_of_Rough_Heston_Models.pdf) | Micro-to-macro scaling from Hawkes to rough fractional Heston diffusions; inspires our formal mathematical axiomatic framework. |
| 2016 | Thibault Jaisson, Mathieu Rosenbaum | **Rough Fractional Diffusions as Scaling Limits of Nearly Unstable Heavy Tailed Hawkes Processes** | *Annals of Applied Probability* | [1504.03100](file:///E:/Financial/literature/2016_1504.03100_Rough_Fractional_Diffusions_as_Scaling_Limits_of_Near.pdf) | Proves heavy-tailed nearly-unstable Hawkes processes scale to rough fractional diffusions ($H = \alpha - 0.5$). Foundations of our fractional stability limits. |
| 2015 | Emmanuel Bacry, Iacopo Mastromatteo, Jean-François Muzy | **Hawkes Processes in Finance** | *Market Microstructure and Liquidity* | [1502.04592](file:///E:/Financial/literature/2015_1502.04592_Hawkes_Processes_in_Finance.pdf) | The definitive review of Hawkes processes in market microstructure, providing classical benchmark specifications and signature plot theory. |
| 2013 | Stephen J. Hardiman, Nicolas Bercot, Jean-Philippe Bouchaud | **Reflected Hawkes Processes and 'Critical' Market Reflexivity** | *European Physical Journal B* | [1302.1405](file:///E:/Financial/literature/2013_1302.1405_Reflected_Hawkes_Processes_and_Critical_Market_Reflexivity.pdf) | Discovers market near-criticality ($n^* \to 1$) and power-law memory $t^{-(1+\delta)}$ in order books. |

---

## Comparative Architectural Matrix: ML-TFHP vs Literature

| Dimension | Classical Hawkes (Bacry 2015) | Power-Law Hawkes (Hardiman 2013) | Fractional Hawkes (Habyarimana 2023) | Tempered ML HP (Gupta 2024) | **ML-TFHP (This Research)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Memory Kernel $\phi(t)$** | $\eta e^{-\beta t}$ | $\eta(t+c)^{-(1+\delta)}$ | $t^{\alpha-1} E_{\alpha,\alpha}(-\beta^\alpha t^\alpha)$ | $e^{-\nu t} t^{\beta-1} E_{\beta,\beta}(-c t^\beta)$ | $\mathbf{\eta e^{-\gamma t} t^{\alpha-1} E_{\alpha,\alpha}(-\beta^\alpha t^\alpha)}$ |
| **Singularity at $t \to 0$** | Finite | Singularity (needs $c$) | Integrable $\sim t^{\alpha-1}$ | Integrable $\sim t^{\beta-1}$ | **Integrable $\sim t^{\alpha-1}$ (No ad-hoc $c$)** |
| **Long-Memory Tempering** | Exponential only | Pure power law | Pure power law | Exponential cut-off | **Continuous Spectral Mixture** |
| **Closed-Form Stationarity** | $\eta / \beta < 1$ | No closed form | Unbounded if $\beta \le 1$ | Analytical Laplace | **$n^* = \frac{\eta}{\gamma^\alpha + \beta^\alpha} < 1$ (Exact)** |
| **State-Space Likelihood** | $\mathcal{O}(N)$ | $\mathcal{O}(N^2)$ (Bottleneck) | $\mathcal{O}(N^2)$ (Intractable) | None (Pure theory) | **$\mathcal{O}(NM)$ (Markovian Quadrature)** |
| **Multivariate System** | Yes ($D \times D$) | No ($\mathcal{O}(D^2 N^2)$) | No (Univariate only) | No (Univariate only) | **Bivariate Asymmetric JIT Engine** |
| **Multi-Horizon Forecasting** | Exponential decay | Heuristic | Laplace inversion | Asymptotic limits | **Exact Closed-Form Renewal Matrix** |
| **Empirical Market Data** | Equities / Futures | Futures | Simulation only | Simulation only | **100,000 Binance BTCUSDT Trades** |
| **FRTB Basel Backtesting** | Fails clustering | None | None | None | **Kupiec + Christoffersen + Acerbi-Székely** |
