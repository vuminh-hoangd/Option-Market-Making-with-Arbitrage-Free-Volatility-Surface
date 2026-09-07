# Option Market Making with Arbitrage-Free Volatility Surface



## No arbitrage eSSVI volatility surface




## Option Market Making and Volatility Arbitrage

Given maturity $\tau$ and strike price $K$, the maker thinks realised will beat implied — they want to be **long** vol, with half a trading day $T$. 
Market-making takes place over a time interval $[0,T]$, with $T\in (0,\tau).$ 

**Objective.** The quotes are not chosen to maximize expected P&L; they
maximize P&L net of the cost of carrying inventory the whole time:

$$\mathbb{E}\Big[V_T - \beta\int_0^T Q_u^2du - \alpha Q_T^2 \Big]$$

The optimal bid/ask half-spreads (eq. 11) are:

$$\delta^{b,*}(t,s,q) = \underbrace{\frac{1}{\kappa^b}}_{\text{liquidity}} - \underbrace{\psi_1(t,s)}_{\text{vol-arb edge + order-flow}} - \underbrace{(2q+1)\psi_2(t)}_{\text{inventory control}}$$

$$\delta^{a,*}(t,s,q) = \underbrace{\frac{1}{\kappa^a}}_{\text{liquidity}} + \underbrace{\psi_1(t,s)}_{\text{vol-arb edge + order-flow}} + \underbrace{(2q-1)\psi_2(t)}_{\text{inventory control}}$$







## Results

### SVI and eSSVI

**Per-expiry SVI fit against the live market smile:**

![SVI fit vs. market smile](pics/SVI-fit.png)

**Global eSSVI surface fit across all expiries at once:**

![Global eSSVI surface](pics/eSSVI-raw.png)


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


