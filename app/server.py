"""Local web UI for the heatablate ablation model.

Serves app/static/ and exposes POST /api/run, which runs the real
AblationFront1D solver for one case (and optionally a multi-station nose
map) and returns JSON. Standard library only, plus numpy via heatablate.

    python app/server.py [port]        # default http://127.0.0.1:8000
"""
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatablate import AblationFront1D, ConvectiveRadiativeBC, Material  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_POINTS = 600  # time-series samples returned to the browser
MAX_WORK = 4_000_000  # cap on (nodes x steps x runs) so a request stays responsive
MAX_STATIONS = 12


class BadRequest(ValueError):
    pass


def num(d, key, lo, hi, default=None, what=None):
    v = d.get(key, default)
    if v is None or isinstance(v, bool):
        raise BadRequest(f"{what or key} is required")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise BadRequest(f"{what or key} must be a number")
    if not math.isfinite(v) or v < lo or v > hi:
        raise BadRequest(f"{what or key} must be between {lo:g} and {hi:g}")
    return v


def build_material(m):
    eps = num(m, "emissivity", 0.0, 1.0, what="emissivity")
    k = num(m, "k", 1e-3, 1e3, what="conductivity k")
    rho = num(m, "rho", 1.0, 3e4, what="density rho")
    if m.get("mode") == "diffusivity":
        alpha = num(m, "alpha", 1e-9, 1e-2, what="diffusivity alpha [m^2/s]")
        cp = k / (alpha * rho)
    else:
        cp = num(m, "cp", 10.0, 1e5, what="specific heat cp")
        alpha = k / (rho * cp)
    kw = {}
    if m.get("ablation_enabled", True):
        kw["ablation_temperature"] = num(m, "ablation_temperature", 300.0, 1e4, what="ablation temperature")
        kw["heat_of_ablation"] = num(m, "heat_of_ablation", 1e4, 1e9, what="heat of ablation")
    mat = Material(k=k, rho=rho, cp=cp, emissivity=eps, **kw)
    return mat, alpha, cp


def build_h(hs, t_end):
    shape = hs.get("shape", "hump")
    if shape == "hump":
        hp = num(hs, "h_peak", 0.0, 1e6, what="peak h")
        tp = num(hs, "t_peak", 1e-3, 1e5, what="time of peak")

        def h(t):
            tau = float(t) / tp
            return hp * tau * math.exp(1.0 - tau)
    elif shape == "constant":
        hc = num(hs, "h_peak", 0.0, 1e6, what="h")

        def h(t):
            return hc
    elif shape == "table":
        rows = hs.get("table") or []
        if len(rows) < 2:
            raise BadRequest("h(t) table needs at least two (t, h) rows")
        try:
            ts = np.array([float(r[0]) for r in rows])
            hv = np.array([float(r[1]) for r in rows])
        except (TypeError, ValueError, IndexError):
            raise BadRequest("h(t) table rows must be (t, h) number pairs")
        if not (np.all(np.isfinite(ts)) and np.all(np.isfinite(hv))) or np.any(np.diff(ts) <= 0) or np.any(hv < 0):
            raise BadRequest("h(t) table: times must increase, h must be >= 0")

        def h(t):
            return float(np.interp(t, ts, hv))
    else:
        raise BadRequest("unknown heating shape")
    return h


def downsample(a, n=MAX_POINTS):
    a = np.asarray(a)
    if len(a) <= n:
        return a
    idx = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    return a[idx]


def run_case(req):
    mat, alpha, cp = build_material(req.get("material", {}))
    g = req.get("geometry", {})
    L = num(g, "thickness_mm", 0.1, 1000.0, what="thickness [mm]") * 1e-3
    n = int(num(g, "n_nodes", 21, 801, what="nodes"))
    dt = num(g, "dt", 1e-4, 10.0, what="time step [s]")
    t_end = num(g, "t_end", 0.01, 1e5, what="duration [s]")
    T0 = num(g, "T0", 1.0, 1e4, what="initial temperature")
    hs = req.get("heating", {})
    T_aw = num(hs, "T_aw", 1.0, 1e5, what="adiabatic wall temperature")
    T_inf = num(hs, "T_inf", 0.0, 1e4, 0.0, what="background temperature")
    h = build_h(hs, t_end)

    stations = req.get("stations") or []
    if len(stations) > MAX_STATIONS:
        raise BadRequest(f"at most {MAX_STATIONS} stations")
    phis = [num({"p": p}, "p", 0.0, 89.0, what="station angle") for p in stations]

    n_steps = int(round(t_end / dt))
    if n_steps < 1:
        raise BadRequest("duration must be at least one time step")
    if n * n_steps * (1 + len(phis)) > MAX_WORK:
        raise BadRequest(
            f"case too large ({n} nodes x {n_steps} steps x {1 + len(phis)} runs); "
            "use fewer nodes, a larger time step, a shorter duration, or fewer stations"
        )

    def solve(scale):
        # Safety net: the surface Newton iteration is robust, but if the
        # solver ever produces non-finite output, tell the user instead of
        # returning garbage.
        with np.errstate(all="ignore"):
            res = _solve(scale)
        if not (np.all(np.isfinite(res["T_back"])) and np.all(np.isfinite(res["s"]))):
            raise BadRequest(
                "The solver produced non-finite temperatures. Try more nodes and/or a smaller time step "
                "(Fine resolution), and check that the inputs are physically reasonable."
            )
        return res

    def _solve(scale):
        bc = ConvectiveRadiativeBC(h=lambda t: scale * h(t), T_aw=T_aw, emissivity=mat.emissivity, T_inf=T_inf)
        return AblationFront1D(mat, L, n, bc, theta=1.0).solve(np.full(n, T0), dt, t_end)

    r = solve(1.0)
    t, s, Tb = r["t"], r["s"], r["T_back"]
    ablated = bool(s[-1] > 0)
    onset = float(t[int(np.argmax(s > 0))]) if ablated else None
    eb = r["energy_balance"]

    warnings = []
    if abs(eb["residual_pct"]) > 5.0:
        warnings.append(
            f"Energy balance is off by {eb['residual_pct']:.1f}% of incident energy, so this result is not "
            "trustworthy. Increase the number of nodes and/or reduce the time step and re-run."
        )
    out = {
        "warnings": warnings,
        "properties": {"alpha": alpha, "cp": cp, "k": mat.k, "rho": mat.rho},
        "t": downsample(t).tolist(),
        "h": [h(x) for x in downsample(t)],
        "s_mm": (downsample(s) * 1e3).tolist(),
        "T_back": downsample(Tb).tolist(),
        "x_mm": ((r["x_final"] - r["x_final"][0]) * 1e3).tolist(),
        "T_final": r["T_final"].tolist(),
        "summary": {
            "ablated": ablated,
            "onset_s": onset,
            "recession_mm": float(s[-1] * 1e3),
            "remaining_mm": float((L - s[-1]) * 1e3),
            "T_back_final": float(Tb[-1]),
            "T_back_peak": float(Tb.max()),
            "T_surface_final": float(r["T_final"][0]),
            "residual_pct": float(eb["residual_pct"]),
            "E_in": float(eb["E_in"]),
        },
    }
    if phis:
        rows = []
        for phi in phis:
            rr = solve(math.cos(math.radians(phi)))
            rows.append({
                "phi": phi,
                "recession_mm": float(rr["s"][-1] * 1e3),
                "T_back_final": float(rr["T_back"][-1]),
                "T_back_peak": float(rr["T_back"].max()),
            })
        out["stations"] = rows
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_000_000:
                raise BadRequest("request too large")
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise BadRequest("request must be a JSON object")
            self._send(200, run_case(req))
        except BadRequest as e:
            self._send(400, {"error": str(e)})
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
        except Exception as e:  # solver failure: report, don't crash the server
            self._send(500, {"error": f"solver error: {e}"})

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"heatablate UI running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
