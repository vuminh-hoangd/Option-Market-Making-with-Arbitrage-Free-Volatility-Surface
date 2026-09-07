# Option Market Making with Arbitrage-Free Volatility Surface



## No arbitrage eSSVI volatility surface




## Option Market Making and Volatility Arbitrage

For a market maker's own view on realized volatility versus the
market's implied volatility, what bid/ask should they quote, accounting for
inventory risk?

Given a market maker who provides pricing quotes for a European call option with a given strike $K > 0$ and maturity $\tau > 0$, with $`(S_t)_{t \ge 0}`$ being the price process of the underlying risky asset. From the perspective of the market maker, $`(S_t)_{t \ge 0}`$ has the dynamics:
$`\mathrm{d}S_t / S_t = \sigma_t \mathrm{d}B_t`$,
where $B$ is a standard one-dimensional Brownian motion. The market maker does not have any view on the (short-term) asset drift, while $`(\sigma_t)_{t \ge 0}`$ is their subjective assessment of the asset volatility process.

**Objective.** The quotes are not chosen to maximize expected P&L; they
maximize P&L net of the cost of carrying inventory the market making period:

$$\arg\max \mathbb{E}\Big[V_T - \beta\int_0^T Q_u^2du - \alpha Q_T^2 \Big]$$

where $V_T$ is the market maker's total portfolio value at the close, $Q_t$ is the market maker's option inventory at time $t$ (how many lots of the option they're currently holding) and $\alpha, \beta$ are two penalty parameters that controls the risk level at the end of a trading period
and the intraday risk exposure throughout the entire duration of
the market-making, respectively.

The optimal bid/ask half-spreads (eq. 11) are:

$$\delta^{b,*}(t,s,q) = \underbrace{\frac{1}{\kappa^b}}_{\text{liquidity}} - \underbrace{\psi_1(t,s)}_{\text{vol-arb edge + order-flow}} - \underbrace{(2q+1)\psi_2(t)}_{\text{inventory control}}$$

$$\delta^{a,*}(t,s,q) = \underbrace{\frac{1}{\kappa^a}}_{\text{liquidity}} + \underbrace{\psi_1(t,s)}_{\text{vol-arb edge + order-flow}} + \underbrace{(2q-1)\psi_2(t)}_{\text{inventory control}}$$


$\psi_1(t,s) = \varphi(t,s) + \text{order-flow}$ is the maker's edge: $\varphi \propto (\sigma^2 -\sigma^2_{\text{imp}})$
the difference of variances between the maker's believed volatility and the market-implied volatility, with $\sigma^2_{\text{imp}}(K,\tau)$ is sourced from the calibrated eSSVI implied volatility surface, weighted by the option's dollar gamma and discounted by the kernel $D(t,u)\approx e^{-\eta(u-t)}$.
$\psi_2(t) < 0$ is the inventory weight —  
$𝑞$ does not change the total quoted width, only how it's split between bid and ask.







## Results

### SVI and eSSVI

**Per-expiry SVI fit against the live market smile:**

![SVI fit vs. market smile](pics/SVI-fit.png)

**Global eSSVI surface fit across all expiries at once:**

![Global eSSVI surface](pics/eSSVI-raw.png)

**Benchmark all three surfaces:**

| Model | Params | Expiries Fitted | Median \|dvol\| | Mean | P90 | Inside Bid-Ask |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVI + interpolate** | 40 | 8 | 0.150% | 0.618% | 2.112% | 71% |
| **SSVI** | 13 | 10 | 1.012% | 2.057% | 5.389% | 32% |
| **Global eSSVI** | 30 | 10 | 0.156% | 0.348% | 0.523% | 88% |


### Optimal Market Making polity

**Vol-arb quoting policy: terminal PnL and inventory vs. fixed-width quoting baselines:**

![Terminal portfolio value and inventory](pics/PnL-and-inventory.png)


**Benchmark.** Against a "fixed-width" quoter that ignores inventory and
the vol view entirely, the fixed-width rule can earn *more* raw P&L
($1,472 vs $1,431) since it never turns down a favorable trade to manage
risk — but scored on the actual objective above, the optimal rule wins
decisively (it pays $47 in inventory-risk cost for that P&L vs. the
fixed-width rule's $289, finishing $202 ahead net).


