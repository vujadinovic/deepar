# DeepAR: Probabilistic Forecasting of Financial Time Series

A university project (Faculty of Mathematics, University of Belgrade, 2026) by Kristina Sandić, Stefan Vujadinović, Boris Stanković and Marko Milinić.

## The idea

When we forecast a stock price, one number is rarely enough. A single point forecast says where the price is *expected* to be, but says nothing about how far it might move. For risk management it matters more to know the whole distribution of possible outcomes, and especially its lower tail. For this reason, every model in this project produces a **probabilistic forecast**: a set of simulated future paths from which quantiles and intervals are computed.

Classical models are *local*: each series is fitted on its own history. DeepAR takes a different approach. It is a single *global* model, trained on many related series at once, so it can learn patterns that the series share. The question this project asks is simple: **does this added complexity actually pay off?**

## Data and setup

- Daily adjusted close prices for 30 large, liquid US companies from different sectors, 2010–2020. Results are reported for AAPL, GS, NKE and XOM.
- Chronological split. The test period is about 12 trading days in March 2020, the COVID-19 crash. This choice is intentional: models are tested under stress, where reliable risk estimates matter most.
- Classical models work with log returns $r_t = \ln P_t - \ln P_{t-1}$, which are approximately stationary. Prices are rebuilt from simulated returns as $\hat{P}_{t+h} = P_t \exp\left(\sum_{k=1}^{h} r_{t+k}\right)$.

## Models

The baselines are ordered by increasing complexity. Each one removes one limitation of the previous one:

1. **Random walk**: returns are i.i.d. $\mathcal{N}(0, \sigma^2)$. The market is unpredictable.
2. **AR(1)**: $r_t = \varphi_0 + \varphi_1 r_{t-1} + \varepsilon_t$. Adds dependence on the past.
3. **ARIMA(1,0,1)**: adds a moving-average term $\theta_1 \varepsilon_{t-1}$. Models past shocks.
4. **GARCH(1,1)**: $\sigma_t^2 = \omega + \alpha a_{t-1}^2 + \beta \sigma_{t-1}^2$. Models time-varying volatility.
5. **DeepAR** (PyTorch): an LSTM that, at each step, outputs the parameters of a distribution instead of a single value. The likelihood of a future path factorizes as
   $$Q_\Theta(z_{t_0:T} \mid z_{1:t_0-1}) = \prod_{t=t_0}^{T} \ell\big(z_t \mid \theta(h_t, \Theta)\big).$$
   The network is trained by maximizing the log-likelihood over all series, and forecasts are obtained by ancestral sampling. Three likelihoods $\ell$ were tested: Gaussian, Student's $t$ and log-normal.

## Metrics

- **RMSE** and **ND** (normalized deviation) measure the accuracy of the central forecast (the median of the simulated paths).
- **Quantile loss** $\mathrm{QL}_{0.10}$ and $\mathrm{QL}_{0.90}$ measure how well the lower and upper tails of the distribution are estimated. $\mathrm{QL}_{0.10}$ is directly related to Value at Risk.

## Results

Averages over all tickers on the test period (lower is better):

| Model | RMSE | ND | $\mathrm{QL}_{0.10}$ | $\mathrm{QL}_{0.90}$ |
|---|---|---|---|---|
| Random walk | 11.54 | 0.0686 | 0.0600 | **0.0297** |
| AR(1) | **9.50** | **0.0641** | 0.0447 | 0.0377 |
| ARIMA(1,0,1) | 11.30 | 0.0659 | 0.0558 | 0.0305 |
| GARCH(1,1) | 11.37 | 0.0670 | **0.0359** | 0.0472 |
| DeepAR, log-normal | 49.93 | 0.0969 | 0.0576 | 0.0785 |
| DeepAR, Student's $t$ | 32.25 | 0.0710 | 0.0458 | 0.0449 |
| DeepAR, Gaussian | 29.36 | 0.0757 | 0.0623 | 0.0334 |

- AR(1) gives the best central forecast.
- GARCH(1,1) gives the best estimate of the lower (risk) tail, as theory predicts: it is the only model that adapts volatility to the current regime.
- Among the DeepAR variants, Student's $t$ is the best overall. Its heavy tails describe extreme moves better. The Gaussian variant underestimates downside risk.

## Conclusion

On a short and extreme test period, the simpler classical models were more robust than DeepAR. This does not show that deep learning is useless here. It shows that its advantages depend on conditions: DeepAR needs a lot of data and a more representative period to show its strengths. Two broader lessons follow:

1. Accuracy of the center and quality of the tails do not have to go together, so probabilistic evaluation is necessary.
2. A more complex model does not guarantee a better forecast. The method should match the amount of data and the market regime.

A natural next step is to evaluate on longer and more varied test periods, with more careful hyperparameter tuning for DeepAR.

## Reference

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191.
