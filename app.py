"""Streamlit app: Present-pegged societal cycle model for joint-loss estimation."""
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import replace

import core
import sociology
import laws
import data

st.set_page_config(page_title="Social Cycles — Joint-Loss Pricer", layout="wide")
st.title("Present-Pegged Societal Cycle Model for Joint-Loss Estimation")
st.caption("ADB Quantum Challenge — prototype. The app estimates the risk and price of a disaster-insurance contract for two sites. Given hazard data (how often earthquakes and heavy rain occur at each site), contract terms (how much each site is covered for), and a 3-year term, it computes five risk numbers — expected loss, probability both sites pay out, probability total loss exceeds a threshold, expected loss above that threshold, and probability both contracts are completely exhausted — and then optionally prices the contract using a four-law premium decomposition. It does this two ways: by exact enumeration (fast, exact, only works for 2 sites) and by sampling (Monte Carlo, Sobol, stratified, importance — works for any number of sites but has error). It also has a societal-cycle layer that can make the hazard rates non-stationary (year-by-year rather than a fixed annual rate) and a quantum tab that estimates how many quantum queries an ideal amplitude-estimation algorithm would need versus classical Monte Carlo. It is a benchmark-validation tool plus a live-data calculator: the same engine, two sources of input. The two modes are two input sources. Classical core validated against "
           "Table 5; cycle layer is a testable hypothesis, not an established fact.")



# ---------------- sidebar ----------------
mode = st.sidebar.radio("Project type", ["ADB benchmark", "Farm project"])

st.sidebar.subheader("Pricing overlay (not evaluated)")
lp = laws.LawParams(
    Dm=st.sidebar.slider("Dm: min valid payout fraction", 0.0, 1.0, 0.0, .05),
    beta=st.sidebar.slider("beta: liquidity uplift", 0.0, 2.0, .5, .05),
    capital_share=st.sidebar.slider("capital share", 0.0, 1.0, .8, .05),
    effort=st.sidebar.slider("effort E", 0.25, 5.0, 1.0, .25),
    k=st.sidebar.slider("fair-exchange k", 0.0, 1.0, .25, .05),
    gamma=st.sidebar.slider("circulation gamma", 0.5, 1.5, 1.0, .05),
    natural_cap=st.sidebar.slider("natural cap", 0.1, 0.6, .35, .05))

# ---------------- inputs ----------------
if mode == "ADB benchmark":
    scn = st.selectbox("Scenario", core.SCENARIOS)
    p = core.adb_params(scn)
    haz_note = "Parameters from challenge Tables 1-2 (stationary benchmark)."
else:
    st.subheader("Farm project")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Project name", "Pilot farm cluster")
        latA = st.number_input("Site A lat", value=42.87, format="%.4f")
        lonA = st.number_input("Site A lon", value=74.59, format="%.4f")
        eq1 = st.number_input("A: quake limit (USD m)", 0.1, 1e4, 7.0)
        pr1 = st.number_input("A: rain limit (USD m)", 0.1, 1e4, 1.0)
        thrA = st.number_input("A: rain trigger (mm)", 1.0, 1000.0, 60.0)
    with c2:
        crop = st.selectbox("Crop", ["Wheat", "Maize", "Rice", "Orchard", "Other"])
        latB = st.number_input("Site B lat", value=38.56, format="%.4f")
        lonB = st.number_input("Site B lon", value=68.77, format="%.4f")
        eq2 = st.number_input("B: quake limit (USD m)", 0.1, 1e4, 7.0)
        pr2 = st.number_input("B: rain limit (USD m)", 0.1, 1e4, 1.0)
        thrB = st.number_input("B: rain trigger (mm)", 1.0, 1000.0, 60.0)
    d1, d2, d3, d4 = st.columns(4)
    radius = d1.number_input("Quake radius (km)", 10, 500, 100)
    hist_years = d2.number_input("History (years)", 10, 60, 30)
    window = d3.number_input("Rain window (days)", 1, 14, 3)
    term = d4.number_input("Term (years)", 1, 5, 3)
    tiers = st.text_input("Magnitude tiers", "5.0,5.5,6.0,6.5")

    if st.button("Fetch hazard data (USGS + Open-Meteo)"):
        try:
            th = [float(x) for x in tiers.split(",")]
            evA = data.usgs_events(latA, lonA, radius, th[0], hist_years)
            evB = data.usgs_events(latB, lonB, radius, th[0], hist_years)
            l1, l2, c = data.quake_rates(evA, evB, th, hist_years)
            q1, q2, crs, ny = data.rain_trigger(
                data.daily_rain(latA, lonA, hist_years),
                data.daily_rain(latB, lonB, hist_years),
                window, thrA, thrB)
            st.session_state.haz = dict(
                l1=l1, l2=l2, c=c, q1=q1, q2=q2, crs=crs,
                nA=len(evA), nB=len(evB), ny=ny)
            st.success(f"Fetched {len(evA)} A-events, {len(evB)} B-events, "
                       f"{ny} rain-years.")
        except Exception as e:
            st.error(f"Fetch failed ({e}). Using/editing the defaults below.")

    h = st.session_state.get("haz", dict(
        l1=np.array([.016, .005, .006, .012]),
        l2=np.array([.023, .004, .003, .010]),
        c=.003, q1=.02, q2=.02, crs=.25, nA=0, nB=0, ny=0))

    st.markdown("**Hazard parameters** (fetched or editable)")
    tab = pd.DataFrame(
        {"Site A": h["l1"], "Site B": h["l2"]},
        index=["k=1 (25%)", "k=2 (50%)", "k=3 (75%)", "k=4 (100%)"]).T
    tab = st.data_editor(tab, key=f"ed{h['nA']}{h['nB']}{h['c']}",
                         use_container_width=True)
    e1, e2, e3, e4 = st.columns(4)
    cc = e1.number_input("Shared quake rate / yr", 0.0, 1.0, float(h["c"]), format="%.5f")
    qa = e2.number_input("Rain trigger prob A", .001, .9, float(h["q1"]), format="%.4f")
    qb = e3.number_input("Rain trigger prob B", .001, .9, float(h["q2"]), format="%.4f")
    crs = e4.number_input("Shared rain share", 0.0, 1.0, float(h["crs"]))
    p = core.Params(
        tab.loc["Site A"].values.astype(float),
        tab.loc["Site B"].values.astype(float),
        cc, qa, qb, 1.0, crs, (eq1, eq2), (pr1, pr2),
        int(term), T=max(eq1 + pr1, eq2 + pr2))
    haz_note = f"{name} / {crop}. Fetch sample: {h['ny']} rain-years."

# ---------------- compute ----------------
E = core.exact(p)["value"]
P = laws.price(p, lp)

tabs = st.tabs(["Results", "Society Cycle", "Benchmark", "Quantum",
                "Compare", "Pricing"])

# ---- Results ----
with tabs[0]:
    st.caption(haz_note)
    a, b, c_, d = st.columns(4)
    a.metric("Expected principal loss (USD m)", f"{E[0]:.3f}")
    b.metric("P(both units pay)", f"{E[1]*100:.3f}%")
    c_.metric("P(loss > T)", f"{E[2]*100:.3f}%")
    d.metric("Stop loss (USD m)", f"{E[3]:.3f}")
    st.dataframe(pd.DataFrame({"target": core.TARGETS, "exact": E}),
                 hide_index=True)

    st.markdown("**Sampling vs exact at a fixed budget**")
    n = st.number_input("Path budget", 1000, 2_000_000, 20000, step=1000,
                        key="n_results")
    engs = st.multiselect(
        "Engines",
        [x for x in core.ENGINES if x != "exact"],
        ["Monte Carlo", "Stratified", "Importance"])
    rows = []
    for en in engs:
        r = core.ENGINES[en](p, int(n), np.random.default_rng(0))
        rows.append(pd.DataFrame({
            "engine": en, "target": core.TARGETS,
            "estimate": r["value"],
            "95% half-width": r["ci"],
            "abs error": np.abs(r["value"] - E),
            "queries": r["queries"]}))
    if rows:
        st.dataframe(pd.concat(rows), hide_index=True)

# ---- Society Cycle ----
with tabs[1]:
    st.markdown("### Present-pegged cycle fit")
    st.caption("Model is pegged at (t_now, phi_now). T* is fitted by span search. "
               "Held-out accuracy and permutation p-value are the evidence. "
               "This does NOT reproduce the stationary benchmark; it is an extension.")

    t_now = st.number_input("Present year", 2020, 2030, 2024)
    phi_now = st.selectbox(
        "Present phase (from independent classifier — pre-register this)",
        sociology.PHASES, index=0)

    country = st.selectbox("Country history", ["KG", "TJ"])

    lam_base = st.session_state.get("lam_base", dict(sociology.DEFAULT_LAM_BASE))
    sent_mult = st.session_state.get("sent_mult", dict(sociology.DEFAULT_SENT_MULT))

    if st.button("Fit cycle (span search + held-out test)"):
        try:
            hist = sociology.load_history(country)
            fit = sociology.fit_cycle(
                int(t_now), phi_now, hist,
                sociology.EVENT_PHASE_SETS,
                perm_reps=300, seed=0)
            st.session_state.fit = fit
            st.session_state.fit_country = country
        except FileNotFoundError:
            st.error(f"History file not found: data/history_{country}.csv. "
                     "Upload a CSV with columns [year, event, disaster].")
        except Exception as e:
            st.error(f"Fit failed: {e}")

    if "fit" in st.session_state:
        fit = st.session_state.fit
        a, b, c_ = st.columns(3)
        a.metric("Best-fit T* (years)", f"{fit.T_star:.1f}")
        b.metric("Held-out accuracy",
                 f"{fit.heldout_acc:.2%}" if fit.heldout_acc is not None else "—")
        c_.metric("Permutation p-value",
                  f"{fit.heldout_p:.4f}" if fit.heldout_p is not None else "—")
        if fit.heldout_p is not None:
            if fit.heldout_p < 0.05:
                st.success("Hypothesis supported at p < 0.05. Cycle layer is a "
                           "justified non-stationary extension.")
            else:
                st.warning("Null result (p ≥ 0.05). Report honestly; stationary "
                           "benchmark remains the scoreboard.")

        st.markdown("**Score profile over T** (flat = no discriminating power)")
        prof = pd.DataFrame({"T": fit.T_grid, "score": fit.score_profile})
        st.line_chart(prof.set_index("T"))

        st.markdown("**Backward-extended phase sequence (last 200 years)**")
        yrs = np.arange(int(t_now) - 200, int(t_now) + 1)
        st.dataframe(pd.DataFrame({
            "year": yrs,
            "phase": sociology.phase_sequence(yrs, fit)}),
            hide_index=True, height=200)

        st.markdown("**Forward disaster targets under the fitted cycle**")
        sents = [st.selectbox(f"Sentiment year +{k+1}", ['+', '-'], 0)
                 for k in range(int(p.years))]
        E_stat = core.exact(p)["value"]
        try:
            E_cyc = sociology.targets_forward(p, fit, sents,
                                              lam_base, sent_mult)
            st.dataframe(pd.DataFrame({
                "target": core.TARGETS,
                "stationary (benchmark)": E_stat,
                "cycle forward (extension)": E_cyc,
                "delta": E_cyc - E_stat}),
                hide_index=True)
        except Exception as e:
            st.error(f"Forward projection failed: {e}")

        st.markdown("**Sensitivity to phi_now (the model's biggest free choice)**")
        rows = []
        for ph in sociology.PHASES:
            try:
                hist = sociology.load_history(st.session_state.get("fit_country", country))
                f2 = sociology.fit_cycle(int(t_now), ph, hist,
                                         sociology.EVENT_PHASE_SETS,
                                         perm_reps=100, seed=0)
                rows.append({"phi_now": ph, "T*": f2.T_star,
                             "held-out acc": f2.heldout_acc,
                             "p": f2.heldout_p})
            
            except Exception:
                rows.append({"phi_now": ph, "T*": np.nan,
                             "held-out acc": np.nan, "p": np.nan})

        sens = pd.DataFrame(rows)
        st.dataframe(
            sens.style.format({
                "T*": "{:.2f}",
                "held-out acc": "{:.2%}",
                "p": "{:.4f}",
            }, na_rep="—"),
            hide_index=True,
        )

# ---- Benchmark ----
with tabs[2]:
    if mode == "ADB benchmark":
        st.markdown("**Exact reference values (challenge Table 5)**")
        st.dataframe(pd.DataFrame([
            {"scenario": s,
             **dict(zip(core.TARGETS,
                        core.exact(core.adb_params(s))["value"]))}
            for s in core.SCENARIOS]), hide_index=True)

    c1, c2, c3 = st.columns(3)
    bn = c1.number_input("Budget (paths)", 1000, 500000, 20000, step=1000,
                         key="bn")
    reps = c2.number_input("Repetitions", 5, 200, 30)
    be = c3.multiselect(
        "Engines",
        [x for x in core.ENGINES if x != "exact"],
        ["Monte Carlo", "Scrambled Sobol", "Stratified", "Importance"])
    if st.button("Run RMSE benchmark"):
        truth, rows = core.benchmark(p, be, int(bn), int(reps))
        df = pd.DataFrame(rows).set_index("engine")
        st.markdown("Relative RMSE vs exact (lower = better). "
                    "Rare-tail target is the last column.")
        st.dataframe(df.style.format("{:.3f}"))
        st.bar_chart(df[core.TARGETS[-1]])

# ---- Quantum ----
with tabs[3]:
    st.markdown("**Oracle-call estimate: MC vs ideal amplitude estimation**")
    tg = st.selectbox("Target", core.TARGETS[1:3] + core.TARGETS[4:], index=2)
    prob = float(E[core.TARGETS.index(tg)])
    rel = st.select_slider("Relative error", [.5, .2, .1, .05, .01], .1)
    if prob > 0:
        nmc, mq = core.qae_vs_mc(prob, rel)
        st.write(f"p = {prob:.2e}: MC needs ~{nmc:,.0f} samples; "
                 f"ideal QAE ~{mq:,.0f} Grover iterations.")
    st.markdown("---")
    st.markdown("""**Resource estimate (two units, IQAE on the 5 targets):**

| Register | Qubits |
|----------|--------|
| 3 × year loader (7 each) | 21 |
| capped accumulator | 8 |
| loss register | ~7 |
| objective + ancilla | ~5 |
| **Total** | **~40–45** |

**Cycle layer impact:** year-indexed loader (3× depth), per-year state space
unchanged at 100 (7 qubits) because phase is deterministic given (t, T*).
Oracle unchanged; only loader depth differs. No quantum code shipped in Phase 1.
""")

# ---- Compare ----
with tabs[4]:
    st.markdown("### Plain benchmark vs cycle-refined")
    st.caption("The five challenge targets must be IDENTICAL to the plain "
               "benchmark — the cycle layer only changes p(x) for the forward "
               "extension scenario, not for the stationary benchmark.")
    plain = dict(zip(core.TARGETS, E))
    cyc = dict(zip(core.TARGETS, E))  # stationary = benchmark
    if "fit" in st.session_state:
        try:
            sents = ['+'] * int(p.years)
            E_cyc = sociology.targets_forward(
                p, st.session_state.fit, sents,
                sociology.DEFAULT_LAM_BASE, sociology.DEFAULT_SENT_MULT)
            cyc = dict(zip(core.TARGETS, E_cyc))
        except Exception:
            pass
    cmp = pd.DataFrame({"plain (benchmark)": plain,
                        "cycle forward (extension)": cyc})
    cmp["delta"] = cmp.iloc[:, 1] - cmp.iloc[:, 0]
    st.dataframe(cmp.style.format("{:.6f}"))

    st.markdown("**Dependence diagnostic** — E[L] identical, joints move")
    rows = []
    for sname in ["adb-independent", "adb-base", "adb-common-shock"]:
        ev = core.exact(core.adb_params(sname))["value"]
        rows.append({"scenario": sname,
                     "E[L] (USD m)": ev[0],
                     "P(both pay) %": ev[1] * 100,
                     "P(both exhausted) %": ev[4] * 100})

    diag = pd.DataFrame(rows)
    st.dataframe(
        diag.style.format({
            "E[L] (USD m)": "{:.4f}",
            "P(both pay) %": "{:.4f}",
            "P(both exhausted) %": "{:.6f}",
        }, na_rep="—"),
        hide_index=True,
    )

    if abs(diag.iloc[0]["E[L] (USD m)"] - diag.iloc[2]["E[L] (USD m)"]) < 1e-9:
        st.success("E[L] identical across independence / base / common-shock. "
                   "Dependence moves joint targets only, as the challenge predicts.")

# ---- Pricing overlay ----
with tabs[5]:
    st.markdown("### Pricing overlay (NOT part of the accuracy claim)")
    st.warning("Deleting this layer would not change the five challenge targets. "
               "Parameters are uncalibrated; treat as a sensitivity study.")
    a, b, c_, d = st.columns(4)
    a.metric("Nominal (L1)", f"{P['nominal']:.3f}")
    b.metric("Real (L2)", f"{P['real']:.3f}")
    c_.metric("Market (L4)", f"{P['market']:.3f}")
    d.metric("Premium", f"{P['premium']:.3f}")
    st.write(f"Shares nominal/real/market: {np.round(P['shares'], 3)}")
    st.write(f"Rate on line: {P['rate_on_line']*100:.2f}%")
    st.write(f"Chain-valid probability: {P['chain_valid_prob']*100:.3f}%")
    st.write("Balanced." if P["balanced"]
             else "**Imbalance: market share exceeds natural cap.**")