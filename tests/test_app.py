import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import server  # noqa: E402


def case(**over):
    req = {
        "material": {"mode": "props", "k": 0.5, "rho": 1400, "cp": 1200, "emissivity": 0.85,
                     "ablation_temperature": 2200, "heat_of_ablation": 8e6},
        "geometry": {"thickness_mm": 18, "n_nodes": 51, "dt": 0.1, "t_end": 120, "T0": 300},
        "heating": {"shape": "hump", "h_peak": 800, "t_peak": 30, "T_aw": 6000, "T_inf": 0},
    }
    for k, v in over.items():
        req[k].update(v) if isinstance(v, dict) else req.__setitem__(k, v)
    return req


def test_run_case_ablates_and_conserves_energy():
    out = server.run_case(case())
    sm = out["summary"]
    assert sm["ablated"] and sm["recession_mm"] > 1.0
    assert abs(sm["residual_pct"]) < 3.0


def test_diffusivity_mode_matches_props_mode():
    a = server.run_case(case())
    alpha = 0.5 / (1400 * 1200)
    b = server.run_case(case(material={"mode": "diffusivity", "alpha": alpha}))
    assert b["summary"]["recession_mm"] == pytest.approx(a["summary"]["recession_mm"], rel=1e-9)


def test_stations_recede_less_off_axis():
    out = server.run_case(case(stations=[0, 45, 70]))
    rec = [s["recession_mm"] for s in out["stations"]]
    assert rec[0] > rec[1] > rec[2]
    assert rec[0] == pytest.approx(out["summary"]["recession_mm"])


@pytest.mark.parametrize("bad", [
    {"material": {"k": -1}},
    {"geometry": {"n_nodes": 5}},
    {"heating": {"shape": "table", "table": [[0, 1]]}},
    {"geometry": {"n_nodes": 801, "dt": 0.0001, "t_end": 1000}},
])
def test_bad_input_rejected(bad):
    with pytest.raises(server.BadRequest):
        server.run_case(case(**bad))


def test_coarse_grid_that_used_to_diverge_now_converges():
    # 41 nodes / dt = 1 s made the old Picard boundary iteration blow up to NaN.
    coarse = server.run_case(case(geometry={"n_nodes": 41, "dt": 1.0}))["summary"]
    assert coarse["recession_mm"] == pytest.approx(6.3235, rel=0.03)  # fine-grid reference
    assert coarse["T_back_final"] == pytest.approx(573.5, rel=0.06)


def _bad_solve(monkeypatch, **fields):
    import numpy as np

    def solve(self, T_initial, dt, t_end, **kw):
        n = int(round(t_end / dt)) + 1
        out = {"t": np.linspace(0, t_end, n), "s": np.zeros(n), "T_back": np.full(n, 300.0),
               "x_final": np.linspace(0, self.length, self.n_nodes), "T_final": np.full(self.n_nodes, 300.0),
               "energy_balance": {"E_in": 1.0, "residual_pct": 0.0}}
        out.update(fields)
        return out

    monkeypatch.setattr(server.AblationFront1D, "solve", solve)


def test_nonfinite_solver_output_reports_error_not_nan(monkeypatch):
    import numpy as np
    _bad_solve(monkeypatch, T_back=np.full(121, np.nan))
    with pytest.raises(server.BadRequest, match="non-finite"):
        server.run_case(case(geometry={"n_nodes": 51, "dt": 1.0}))


def test_poor_energy_balance_is_flagged(monkeypatch):
    _bad_solve(monkeypatch, energy_balance={"E_in": 1.0, "residual_pct": 12.0})
    out = server.run_case(case(geometry={"n_nodes": 51, "dt": 1.0}))
    assert out["warnings"]
