#!/usr/bin/env python3
"""PCBGenius auto-fix efficacy sweep (CPU, no GPU).

Seed = the same deterministic in-scope fault classes the auto-fixer is built to
repair, plus a no-regression set. Metrics are HONEST:
  - fix_rate      = in-scope defects that auto_fix turns structural-FAIL -> PASS
                     divided by in-scope defects attempted.
  - no_false_fix  = healthy / already-PASS netlists that auto_fix must NOT break
                     (real stress parseables + clean golden fixtures). A regression
                     here (PASS -> FAIL) is a bug, so we report it explicitly.
  - Each in-scope case is gated on the DETERMINISTIC structural gate, not ""fixed""
                     alone (fixed=True with a still-FAILING gate is not a fix).

Emit: human table to stdout + JSON at artifacts/autofix_sweep.json (for R2 upload).
Runs anywhere with python3 stdlib + the model/ package (no numpy/scipy/kicad).
"""
import copy
import importlib.util
import io
import json
import os
import sys

REPO_L = "C:\\Users\\John Doe\\Desktop\\PCBGenius_Local"
# cwd is the repo root when run from GH (workflow cd's there) or locally.
_cwd = os.getcwd()

sys.path.insert(0, _cwd)

from model.verification import verify_harness as _vh
from model.verification import auto_fix as _af
from model.verification.test_auto_fix import (  # noqa: E402  (fixture builders)
    _good_led_netlist, _ntc_netlist, _buck_netlist, _mppt_netlist)

DP = {"vin": 12}


def run_all(nl, prompt=None):
    try:
        return _vh.run_all(nl, prompt=prompt)
    except Exception as e:  # fail-closed: never let one case kill the sweep
        return [type("GR", (), {"name": "run_all.ERR", "verdict": "INDETERMINATE", "details": str(e)[:80]})()]


def gate_verdict(nl, gatename, prompt=None):
    for r in run_all(nl, prompt):
        if getattr(r, "name", "") == gatename:
            return getattr(r, "verdict", "INDETERMINATE")
    return "NOT_RUN"


def erc_verdict(nl):
    return gate_verdict(nl, "erc.structural")


def _snap(nl):
    return json.dumps(nl, sort_keys=True)  # cheap regression check (order-insensitive keys)


# --- build in-scope defect cases -------------------------------------------------
def _rev(nl, ref, pin):
    for c in nl.get("components", []):
        if c["ref"] == ref:
            for p_ in c["pins"]:
                if p_["name"] == pin:
                    p_["net"] = "VIN"
    return nl


def _drop(nl, ref):
    nl = copy.deepcopy(nl)
    nl["components"] = [c for c in nl["components"] if c["ref"] != ref]
    return nl


def _led_anode_to_rail():
    nl = _good_led_netlist()
    nl = _drop(nl, "R_LED")
    led = [c for c in nl["components"] if c["ref"] == "LED1"][0]
    for p_ in led["pins"]:
        if p_["name"] == "A":
            p_["net"] = "VIN"
    return nl


def _led_polarity_reversed():
    nl = _good_led_netlist()
    return _rev(nl, "LED1", "K")


def _ntc_coarse():
    return _ntc_netlist("5%")   # >1% coarse -> T_SENS_01 FAIL


def _buck_coarse():
    return _buck_netlist("5%")  # Buck-02 coarse divider -> FAIL


# MPPT footprint repair case (from test_auto_fix)
def _mppt_bad_footprint():
    nl = _mppt_netlist()
    for c in nl.get("components", []):
        if c.get("type") == "bootcon_3" or c.get("ref", "").startswith("MPPT"):
            c["package"] = c.get("package", "0805")
    return nl


IN_SCOPE = [
    ("led.missing_series_r",  _led_anode_to_rail,                 "led.current_limit"),
    ("led.polarity_reversed", _led_polarity_reversed,             "led.current_limit"),
    ("ntc.coarse_pullup",     _ntc_coarse,                        "ntc.pullup"),
    ("buck02.coarse_divider", _buck_coarse,                       "buck02.tolerance"),
    ("mppt.bad_footprint",    _mppt_bad_footprint,                "mppt.check"),
]

# --- no-regression (healthy) set -------------------------------------------------
HEALTHY = [
    ("led.good",   _good_led_netlist),
    ("ntc.fine",   lambda: _ntc_netlist("0.5%")),
    ("buck.fine",  lambda: _buck_netlist("1%")),
]

# real model outputs that already structurally PASS (from stress_result.jsonl)
def _stress_parseables():
    out = []
    p = os.path.join(_cwd, "artifacts", "eval", "stress_result.jsonl")
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                nl = json.loads(r["raw"])
            except Exception:
                continue
            if isinstance(nl, dict):
                out.append(nl)
    return out


def main() -> int:
    results = []

    # -- in-scope: defect must be gated FAIL; measure BOTH repair entries --
    #   public  : auto_fix()          (what capture.py / regenerate.py call)
    #   phase1  : auto_fix_phase1()   (the repair rules the unit tests validate)
    fixed_pub = tried_pub = fixed_p1 = tried_p1 = 0
    for name, builder, gatename in IN_SCOPE:
        nl = builder()
        meta = nl.get("metadata") or {}
        dp = meta.get("design_params") or DP
        v0 = gate_verdict(nl, gatename)
        if v0 != "FAIL":
            results.append({"case": name, "phase": "in_scope", "status": "SKIP",
                            "why": "gate=%s (not a FAIL, nothing to repair)" % v0})
            continue
        tried_pub += 1
        tried_p1 += 1
        before = _snap(nl)
        try:                                    # public entry
            out = _af.auto_fix(nl)
            nl_pub = out.get("netlist") or nl
            v_pub = gate_verdict(nl_pub, gatename)
            if v_pub == "PASS":
                fixed_pub += 1
        except Exception as e:
            v_pub = "ERR:" + str(e)[:20]
        try:                                    # direct phase-1 helper
            o1 = _af.auto_fix_phase1(copy.deepcopy(nl), design_params=dp)
            nl_p1 = o1.get("netlist") or nl
            v_p1 = gate_verdict(nl_p1, gatename)
            if v_p1 == "PASS":
                fixed_p1 += 1
        except Exception as e:
            v_p1 = "ERR:" + str(e)[:20]
        results.append({"case": name, "phase": "in_scope",
                        "gate_before": v0,
                        "public_auto_fix_after": v_pub,
                        "phase1_after": v_p1,
                        "public_fixed": (v_pub == "PASS"),
                        "phase1_fixed": (v_p1 == "PASS")})

    # -- no-regression: healthy + real parseables must stay PASS --
    regress = unchanged = 0
    no_fix_ok = 0
    healthy = HEALTHY + [("stress.seed%d" % i, (lambda n=n: (lambda: n))())
                        for i, n in enumerate(_stress_parseables())]
    # simplified: build healthy list without the lazy-wrapper trick
    healthy = []
    for name, builder in HEALTHY:
        healthy.append((name, builder()))
    for i, n in enumerate(_stress_parseables()):
        healthy.append(("stress.seed%d" % i, n))

    for name, nl in healthy:
        g0 = erc_verdict(nl)
        if g0 != "PASS":
            results.append({"case": name, "phase": "no_regression", "status": "SKIP",
                            "why": "erc=%s (unparseable/no-struct)" % g0}); continue
        before = _snap(nl)
        try:
            out = _af.auto_fix(nl)
            nl2 = out.get("netlist") or nl
        except Exception as e:
            results.append({"case": name, "phase": "no_regression", "status": "ERROR",
                            "why": str(e)[:80]}); continue
        g1 = erc_verdict(nl2)
        if g1 != "PASS":
            regress += 1
            results.append({"case": name, "phase": "no_regression", "status": "REGRESSION",
                            "erc_before": g0, "erc_after": g1})
        else:
            unchanged += 1
            if _snap(nl2) == before and not out.get("fixes"):
                no_fix_ok += 1
            results.append({"case": name, "phase": "no_regression", "status": "OK",
                            "erc": g1, "mutated": (_snap(nl2) != before),
                            "repairs": len(out.get("fixes") or [])})

    rate_pub = (fixed_pub / tried_pub) if tried_pub else 0.0
    rate_p1 = (fixed_p1 / tried_p1) if tried_p1 else 0.0
    no_regress_ok = (len([r for r in results if r["phase"] == "no_regression"]) > 0
                     and regress == 0)
    summary = {
        "in_scope_tried": tried_pub, "in_scope_fixed_public": fixed_pub,
        "public_auto_fix_fix_rate": round(rate_pub, 4),
        "phase1_auto_fix_fix_rate": round(rate_p1, 4),
        "no_regression_total": unchanged + regress,
        "no_regression_clean": unchanged, "no_regression_fail": regress,
        "no_false_fix_preserved": no_fix_ok,
        "rows": results,
    }
    txt = json.dumps(summary, indent=2)
    print(txt)
    try:
        _adir = os.path.join(_cwd, "artifacts")
        os.makedirs(_adir, exist_ok=True)
        with open(os.path.join(_adir, "autofix_sweep.json"), "w", encoding="utf-8") as f:
            f.write(txt)
    except Exception as e:
        print("[sweep] note: could not write artifacts/autofix_sweep.json: %s" % e, file=sys.stderr)
    print()
    print("=== SUMMARY ===")
    print("  public auto_fix()  fix-rate : %d/%d = %.1f%%" % (fixed_pub, tried_pub, rate_pub * 100))
    print("  phase1 auto_fix_phase1()    : %d/%d = %.1f%%" % (fixed_p1, tried_p1, rate_p1 * 100))
    print("  no-regression clean/fail    : %d / %d" % (unchanged, regress))
    return 0


if __name__ == "__main__":
    sys.exit(main())