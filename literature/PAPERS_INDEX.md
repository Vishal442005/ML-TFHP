# Benchmark Literature Index: Fractional Hawkes Processes & Rough Volatility

This directory contains full-text PDFs of the seminal and latest benchmark papers
in the exact mathematical and empirical domain of the Mittag-Leffler Tempered
Fractional Hawkes Process (ML-TFHP).

| Year | Authors | Title | Venue | arXiv / Local PDF | Key Relevance to ML-TFHP |
|:---:|---|---|---|---|---|
| 2016 | Thibault Jaisson, Mathieu Rosenbaum | **Rough Fractional Diffusions as Scaling Limits of Nearly Unstable Heavy Tailed Hawkes Processes** | *Annals of Applied Probability* | [1504.03100](file:///E:/Financial/literature/2016_1504.03100_Rough_Fractional_Diffusions_as_Scaling_Limits_of_Nearly_Unst.pdf) | Proves heavy-tailed nearly-unstable Hawkes processes scale to rough fractional diffusions ($H = \alpha - 0.5$). Foundations of our Caputo fractional formulation. |
| 2019 | Eduardo Abi Jaber, Omar El Euch | **Markovian Approximations of Rough Heston Models** | *SIAM Journal on Financial Mathematics* | [1801.00978](file:///E:/Financial/literature/2019_1801.00978_Markovian_Approximations_of_Rough_Heston_Models.pdf) | Derives Markovian lifting of fractional Volterra kernels via sum of exponentials. Predecessor of our log-uniform quadrature method. |
| 2017 | Pascal Blanc, Jonathan Donier, Jean-Philippe Bouchaud | **Quadratic Hawkes Processes for Financial Volatility** | *Quantitative Finance* | [1609.07474](file:///E:/Financial/literature/2017_1609.07474_Quadratic_Hawkes_Processes_for_Financial_Volatility.pdf) | Formulates quadratic feedback Hawkes models for volatility clustering and leverage effect in financial markets. |
| 2015 | Emmanuel Bacry, Iacopo Mastromatteo, Jean-François Muzy | **Hawkes Processes in Finance** | *Market Microstructure and Liquidity* | [1502.04592](file:///E:/Financial/literature/2015_1502.04592_Hawkes_Processes_in_Finance.pdf) | The definitive review of Hawkes processes in market microstructure, providing standard baseline specifications (EHP). |
| 2013 | Stephen J. Hardiman, Nicolas Bercot, Jean-Philippe Bouchaud | **Reflected Hawkes Processes and 'Critical' Market Reflexivity** | *European Physical Journal B* | [1302.1405](file:///E:/Financial/literature/2013_1302.1405_Reflected_Hawkes_Processes_and_Critical_Market_Reflexivity.pdf) | Discovers market near-criticality ($n^* \to 1$) and power-law memory $t^{-(1+\delta)}$ in order books. |
| 2020 | Donatien Hainaut | **A Fractional Hawkes Process with Mittag-Leffler Intensity** | *Physica A: Statistical Mechanics and its Applications* | [2004.09849](file:///E:/Financial/literature/2020_2004.09849_A_Fractional_Hawkes_Process_with_Mittag_Leffler_Intensity.pdf) | First introduces fractional Hawkes with Mittag-Leffler intensity, but without exponential macro-tempering, closed-form stationarity, or $O(NM)$ quadrature. |
| 2019 | Eduardo Abi Jaber, Martin Larsson, Sergio Pulido | **Affine Volterra Processes** | *Annals of Applied Probability* | [1708.08796](file:///E:/Financial/literature/2019_1708.08796_Affine_Volterra_Processes.pdf) | Rigorous theory of completely monotone kernels, Bernstein's theorem, and infinite-dimensional Markovian representations. |
| 2019 | Omar El Euch, Mathieu Rosenbaum | **The Characteristic Function of Rough Heston Models** | *Mathematical Finance* | [1609.02108](file:///E:/Financial/literature/2019_1609.02108_The_Characteristic_Function_of_Rough_Heston_Models.pdf) | Analytical pricing and fractional Riccati equations under rough volatility ($H \approx 0.1$). |

---

## Comparative Architectural Matrix: ML-TFHP vs Literature

| Dimension | Classical Hawkes (Bacry 2015) | Power-Law Hawkes (Hardiman 2013) | Fractional Hawkes (Hainaut 2020) | **ML-TFHP (This Research)** |
|---|:---:|:---:|:---:|:---:|
| **Memory Kernel** | $\eta e^{-\beta t}$ | $\eta(t+c)^{-(1+\delta)}$ | $t^{\alpha-1} E_{\alpha,\alpha}(-\beta^\alpha t^\alpha)$ | $\mathbf{\eta e^{-\gamma t} t^{\alpha-1} E_{\alpha,\alpha}(-\beta^\alpha t^\alpha)}$ |
| **Singularity at $t \to 0$** | Finite | Singularity (needs $c$) | Integrable $\sim t^{\alpha-1}$ | **Integrable $\sim t^{\alpha-1}$ (No ad-hoc $c$)** |
| **Long-Memory Persistence** | None (Exponential) | Power law | Power law | **Power law + Exponential Tempering** |
| **Closed-Form Stationarity** | $\eta / \beta < 1$ | No closed-form | No closed-form | **$n^* = \frac{\eta}{\gamma^\alpha + \beta^\alpha} < 1$ (Exact)** |
| **Likelihood Complexity** | $\mathcal{O}(N)$ | $\mathcal{O}(N^2)$ (Bottleneck) | $\mathcal{O}(N^2)$ | **$\mathcal{O}(NM)$ (Markovian Quadrature)** |
| **Multi-Horizon Crossover** | Under-forecasts | Not analytical | Not derived | **Exact Matrix Renewal ODE** |
| **FRTB Risk Backtesting** | Fails clustering | Not addressed | None | **Kupiec + Christoffersen + Acerbi-Székely** |
| **Bivariate Microstructure** | Exponential only | $\mathcal{O}(N^2)$ intractability | Univariate only | **$2 \times 2$ Mutual Excitation (Numba JIT)** |
