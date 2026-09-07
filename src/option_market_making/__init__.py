"""
Single-option market-making engine under constant real-world volatility.

Implements the baseline model of Lucic & Tse (2025), "Option market-making
and vol arbitrage" (Risk.net, February 2025), pp. 1-3: the closed-form
Avellaneda-Stoikov / HJB optimal bid-ask quoting rule for one European call,
where the market maker's edge comes from the gap between their own constant
volatility estimate `sigma` and the market-implied `sigma_imp` read off the
calibrated eSSVI surface in `vol_surface`.

Layering, deliberately strict:
- `bs_mm`     Black-Scholes price/Greeks, and the ONLY module that talks to
              the volatility surface.
- `riccati`   psi2(t) and the discount kernel D(t,u) -- pure algebra.
- `edge`      phi_edge(t,s), the volatility-arbitrage edge (one quadrature).
- `psi1`      psi1(t,s), edge plus the asymmetric-order-flow corrections.
- `quoting`   MMParams, optimal spreads (eq. 11) and executable quotes.
- `sim`       Monte Carlo backtest of the quoting rule over [0, T].

Everything below `quoting` is pure math on a frozen `sigma_imp_val`: no
module except `bs_mm` imports the surface, and nothing here hits the network.
"""
