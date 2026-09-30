"""Small compatibility shims for the supported numpy range (>=1.24)."""
import numpy as np

# np.trapezoid was added in numpy 2.0 (np.trapz was deprecated then removed).
trapezoid = getattr(np, "trapezoid", None) or np.trapz
