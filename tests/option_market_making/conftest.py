"""
Shared fixtures: a frozen live Deribit chain, calibrated once to a Global
eSSVI surface and stored in `deribit_snapshot.json`.

The snapshot is replayed offline -- the surface is rebuilt from the stored
eSSVI slice parameters, so these tests exercise the real `implied_vol` query
path against real market data without ever touching the network.
"""
import json
import pathlib

import pytest

from option_market_making.quoting import MMParams
from vol_surface.global_essvi import ESSVISlice, GlobalESSVISurface

SNAPSHOT_PATH = pathlib.Path(__file__).with_name("deribit_snapshot.json")


@pytest.fixture(scope="session")
def snapshot() -> dict:
    with SNAPSHOT_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def surface(snapshot) -> GlobalESSVISurface:
    """The calibrated eSSVI surface from the frozen snapshot."""
    return GlobalESSVISurface(
        maturities=snapshot["maturities"],
        slices=[ESSVISlice(**s) for s in snapshot["slices"]],
        forwards=snapshot["forwards"],
    )


@pytest.fixture(scope="session")
def contract(snapshot) -> dict:
    """The (S0, K, tau) point being quoted, plus its market-implied vol."""
    return {
        "S0": snapshot["spot"],
        "K": snapshot["strike"],
        "tau": snapshot["tau"],
        "sigma_imp": snapshot["sigma_imp"],
    }


@pytest.fixture(scope="session")
def horizon(contract) -> float:
    """
    T = half a trading day, the paper's numerical example. Comfortably inside
    the option's life, which is what the model needs (T < tau).
    """
    return 0.5 / 252.0


@pytest.fixture
def params(contract) -> MMParams:
    """
    A deliberately ASYMMETRIC parameter set, scaled to BTC option prices
    (a ~31-day near-ATM call marks around USD 3.7k here).

    - 1/kappa is ~USD 40 and ~USD 33, i.e. a spread on the order of 1% of the
      option's value;
    - lambda0 corresponds to roughly 100 lots a day a side, with 15% more
      selling pressure than buying (lambda0_b > lambda0_a, so the market maker
      is a net buyer), matching the paper's own net-seller-of-vol setup;
    - alpha and beta are sized so the inventory term moves the quote by a few
      dollars per lot, small against 1/kappa but not negligible;
    - sigma sits ABOVE the implied vol, so the market maker is long vol and the
      edge term is positive.
    """
    return MMParams(
        alpha=2.0,
        beta=4357.0,
        lambda0_b=252.0 * 100.0 * 1.15,
        lambda0_a=252.0 * 100.0,
        kappa_b=0.025,
        kappa_a=0.030,
        sigma=contract["sigma_imp"] + 0.05,
    )


@pytest.fixture
def symmetric_params(params) -> MMParams:
    """Same, but with a perfectly balanced book -- both flow corrections vanish."""
    from dataclasses import replace

    return replace(params, lambda0_b=params.lambda0_a, kappa_b=params.kappa_a)
