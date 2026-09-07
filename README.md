# Volatility Surface Calibration Engine





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


