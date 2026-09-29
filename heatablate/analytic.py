"""Closed-form solutions used to validate the numerical solver.

All three are classical semi-infinite-solid / moving-boundary results.
See docs/PROJECT_PLAN.md for the validation plan and references.
"""
import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import brentq


def semi_infinite_step_temperature(x, t, alpha, T0, Ts):
    """Semi-infinite solid, step change in surface temperature at t=0.

    Surface held at Ts for t>0, initial uniform temperature T0.

        T(x,t) = Ts + (T0 - Ts) * erf( x / (2*sqrt(alpha*t)) )

    x, t: arrays or scalars (t=0 handled as the initial condition)
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(t > 0, x / (2.0 * np.sqrt(alpha * np.maximum(t, 1e-300))), np.inf)
    T = Ts + (T0 - Ts) * erf(eta)
    T = np.where(t > 0, T, T0)
    return T


def semi_infinite_constant_flux_temperature(x, t, alpha, k, q0, T0):
    """Semi-infinite solid, constant surface heat flux q0 applied at t=0.

        T(x,t) - T0 = (q0/k) * [ 2*sqrt(alpha*t/pi) * exp(-x^2/(4*alpha*t))
                                   - x * erfc( x / (2*sqrt(alpha*t)) ) ]
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_at = np.sqrt(alpha * np.maximum(t, 1e-300))
        eta = x / (2.0 * sqrt_at)
        term = 2.0 * np.sqrt(alpha * np.maximum(t, 1e-300) / np.pi) * np.exp(-(x**2) / (4.0 * alpha * np.maximum(t, 1e-300))) - x * erfc(eta)
    T = T0 + (q0 / k) * term
    T = np.where(t > 0, T, T0)
    return T


def stefan_lambda(Ste: float) -> float:
    """Root of the classical one-phase Stefan transcendental equation.

    Front position s(t) = 2*lambda*sqrt(alpha*t). lambda solves:

        lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi)

    Ste = cp*(Ts - Tm) / Lh is the Stefan number (sensible / latent heat),
    with Ts the (fixed) hot surface temperature driving the front and Tm
    the melt/ablation temperature.
    """

    def residual(lam):
        return lam * np.exp(lam**2) * erf(lam) - Ste / np.sqrt(np.pi)

    return brentq(residual, 1e-8, 10.0)


def stefan_front_position(t, alpha, lam):
    """Melt/ablation front position s(t) = 2*lambda*sqrt(alpha*t)."""
    return 2.0 * lam * np.sqrt(alpha * np.asarray(t, dtype=float))


def stefan_temperature(x, t, alpha, Ts, Tm, lam):
    """Temperature in the melted/ablated (liquid) region 0 <= x <= s(t).

        T(x,t) = Ts + (Tm - Ts) * erf(x / (2*sqrt(alpha*t))) / erf(lambda)
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = x / (2.0 * np.sqrt(alpha * np.maximum(t, 1e-300)))
    return Ts + (Tm - Ts) * erf(eta) / erf(lam)


def stefan_front_flux(t, alpha, k, Ts, Tm, lam):
    """Conductive flux arriving at the front from the hot side, which by
    the Stefan energy balance equals rho*Lh*ds/dt:

        q(t) = k*(Ts-Tm)*exp(-lambda^2) / (erf(lambda)*sqrt(pi*alpha*t))

    Used as the driving FluxBC in the front-tracking validation: feeding
    this exact analytic flux into the ablation solver isolates the
    front-recession/remeshing algorithm from the (separately validated,
    Validations 1-2) diffusion scheme and from the nonlinear
    convective/radiative BC used in the full model (Phase 4).
    """
    t = np.asarray(t, dtype=float)
    return k * (Ts - Tm) * np.exp(-(lam**2)) / (erf(lam) * np.sqrt(np.pi * alpha * np.maximum(t, 1e-300)))
