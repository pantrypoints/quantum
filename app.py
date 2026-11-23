import numpy as np, pandas as pd, streamlit as st
from dataclasses import replace
import core, laws, data

st.set_page_config(page_title="Cat-risk joint-loss pricer", layout="wide")
st.title("Joint-loss & premium engine (classical core, Braket-ready)")
st.caption("Two peril-zone units, earthquake + precipitation, 3-year term. Exact enumeration vs sampling baselines, "
           "plus a four-law pricing layer. Law functional forms are modelling assumptions, not calibrated facts.")

# ---------------- sidebar ----------------
mode = st.sidebar.radio("Project type", ["ADB benchmark", "Farm project"])
st.sidebar.subheader("Four-law parameters")
lp = laws.LawParams(
    Dm=st.sidebar.slider("L1 Dm: min valid payout fraction", 0.0, 1.0, 0.0, .05),
    beta=st.sidebar.slider("L2 beta: liquidity value uplift", 0.0, 2.0, .5, .05),
    capital_share=st.sidebar.slider("L2 capital share of exposure", 0.0, 1.0, .8, .05),
    effort=st.sidebar.slider("L2 effort/friction E", 0.25, 5.0, 1.0, .25),
    k=st.sidebar.slider("L4 fair-exchange k", 0.0, 1.0, .25, .05),
    gamma=st.sidebar.slider("L4 circulation gamma", 0.5, 1.5, 1.0, .05),
    natural_cap=st.sidebar.slider("L3 max natural market share", 0.1, 0.6, .35, .05))

# ---------------- inputs ----------------
if mode == "ADB benchmark":
    scn = st.selectbox("Scenario", core.SCENARIOS)
    p = core.adb_params(scn)
    haz_note = "Parameters from the challenge dataset (Tables 1-2)."
else:
    st.subheader("Farm project")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Project name", "Pilot farm cluster")
        latA = st.number_input("Site A lat", value=42.87, format="%.4f"); lonA = st.number_input("Site A lon", value=74.59, format="%.4f")
        eq1 = st.number_input("A: structures/equipment limit (USD m)", 0.1, 1e4, 7.0)
        pr1 = st.number_input("A: crop/rain limit (USD m)", 0.1, 1e4, 1.0)
        thrA = st.number_input("A: rain trigger (mm per window)", 1.0, 1000.0, 60.0)
    with c2:
        crop = st.selectbox("Crop", ["Wheat", "Maize", "Rice", "Orchard", "Other"])
        latB = st.number_input("Site B lat", value=38.56, format="%.4f"); lonB = st.number_input("Site B lon", value=68.77, format="%.4f")
        eq2 = st.number_input("B: structures/equipment limit (USD m)", 0.1, 1e4, 7.0)
        pr2 = st.number_input("B: crop/rain limit (USD m)", 0.1, 1e4, 1.0)
        thrB = st.number_input("B: rain trigger (mm per window)", 1.0, 1000.0, 60.0)
    d1, d2, d3, d4 = st.columns(4)
    radius = d1.number_input("Quake radius (km)", 10, 500, 100)
    hist = d2.number_input("History (years)", 10, 60, 30)
    window = d3.number_input("Rain window (days)", 1, 14, 3)
    term = d4.number_input("Term (years)", 1, 5, 3)
    tiers = st.text_input("Magnitude tiers for 25/50/75/100% payout", "5.0,5.5,6.0,6.5")
    if st.button("Fetch hazard data (USGS + Open-Meteo)"):
        try:
            th = [float(x) for x in tiers.split(",")]
            evA = data.usgs_events(latA, lonA, radius, th[0], hist); evB = data.usgs_events(latB, lonB, radius, th[0], hist)
            l1, l2, c = data.quake_rates(evA, evB, th, hist)
            q1, q2, crs, ny = data.rain_trigger(data.daily_rain(latA, lonA, hist), data.daily_rain(latB, lonB, hist), window, thrA, thrB)
            st.session_state.haz = dict(l1=l1, l2=l2, c=c, q1=q1, q2=q2, crs=crs, nA=len(evA), nB=len(evB), ny=ny)
        except Exception as e:
            st.error(f"Fetch failed ({e}). Using/editing the values below instead.")
    h = st.session_state.get("haz", dict(l1=np.array([.016, .005, .006, .012]), l2=np.array([.023, .004, .003, .010]),
                                         c=.003, q1=.02, q2=.02, crs=.25, nA=0, nB=0, ny=0))
    st.markdown("**Hazard parameters** (fetched or editable; sample defaults until you fetch)")
    tab = pd.DataFrame({"Site A": h["l1"], "Site B": h["l2"]}, index=["k=1 (25%)", "k=2 (50%)", "k=3 (75%)", "k=4 (100%)"]).T
    tab = st.data_editor(tab, key=f"ed{h['nA']}{h['nB']}{h['c']}", width="stretch")
    e1, e2, e3, e4 = st.columns(4)
    cc = e1.number_input("Shared quake rate / yr", 0.0, 1.0, float(h["c"]), format="%.5f")
    qa = e2.number_input("Rain trigger prob A", .001, .9, float(h["q1"]), format="%.4f")
    qb = e3.number_input("Rain trigger prob B", .001, .9, float(h["q2"]), format="%.4f")
    crs = e4.number_input("Shared rain share", 0.0, 1.0, float(h["crs"]))
    p = core.Params(tab.loc["Site A"].values.astype(float), tab.loc["Site B"].values.astype(float), cc, qa, qb, 1.0, crs,
                    (eq1, eq2), (pr1, pr2), int(term), T=max(eq1 + pr1, eq2 + pr2))
    haz_note = f"{name} / {crop}. Shared quake events = same USGS event id inside both radii. Sampling sizes sample years: {h['ny']}."

E = core.exact(p)["value"]
P = laws.price(p, lp)

tabs = st.tabs(["Results", "Four laws", "Benchmark", "Quantum / Braket"])
with tabs[0]:
    st.caption(haz_note)
    a, b, c_, d = st.columns(4)
    a.metric("Expected principal loss (USD m)", f"{E[0]:.3f}")
    b.metric("P(both units pay)", f"{E[1]*100:.3f}%")
    c_.metric("P(loss > T)", f"{E[2]*100:.3f}%")
    d.metric("Premium, 3-law consolidated (USD m)", f"{P['premium']:.3f}")
    st.dataframe(pd.DataFrame({"target": core.TARGETS, "exact": E}), hide_index=True)
    st.markdown("**Sampling vs exact at a fixed budget**")
    n = st.number_input("Path budget", 1000, 2_000_000, 20000, step=1000)
    engs = st.multiselect("Engines", [x for x in core.ENGINES if x != "exact"], ["Monte Carlo", "Stratified", "Importance"])
    rows = []
    for en in engs:
        r = core.ENGINES[en](p, int(n), np.random.default_rng(0))
        rows.append(pd.DataFrame({"engine": en, "target": core.TARGETS, "estimate": r["value"], "95% half-width": r["ci"],
                                  "abs error": np.abs(r["value"] - E), "queries": r["queries"]}))
    if rows: st.dataframe(pd.concat(rows), hide_index=True)

with tabs[1]:
    st.markdown("**Premium decomposition (Law 3 shares of Law 1 + 2 + 4)**")
    a, b, c_, d = st.columns(4)
    a.metric("Nominal (L1)", f"{P['nominal']:.3f}"); b.metric("Real (L2)", f"{P['real']:.3f}")
    c_.metric("Market (L4)", f"{P['market']:.3f}"); d.metric("Rate on line", f"{P['rate_on_line']*100:.2f}%")
    st.write(f"Shares nominal/real/market: {np.round(P['shares'], 3)}. Chain-valid probability (all units pay above Dm): "
             f"{P['chain_valid_prob']*100:.3f}%. " + ("Balanced." if P["balanced"] else
             "**Imbalance: market share exceeds the natural cap, so the price overfeeds the fee layer.**"))
    rows = [{"variant": k, "joint pay %": core.exact(v)["value"][1] * 100, "P(both exhausted) %": core.exact(v)["value"][4] * 100,
             "nominal": (q := laws.price(v, lp))["nominal"], "real": q["real"], "market": q["market"],
             "premium": q["premium"], "market share": q["shares"][2]} for k, v in laws.variants(p).items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Dependence leaves expected loss unchanged but moves joint targets and the value-weighted (L2) component.")

with tabs[2]:
    if mode == "ADB benchmark":
        st.markdown("**Exact reference values (should match challenge Table 5)**")
        st.dataframe(pd.DataFrame([{"scenario": s, **dict(zip(core.TARGETS, core.exact(core.adb_params(s))["value"]))}
                                   for s in core.SCENARIOS]), hide_index=True)
    c1, c2, c3 = st.columns(3)
    bn = c1.number_input("Budget (paths)", 1000, 500000, 20000, step=1000, key="bn")
    reps = c2.number_input("Repetitions", 5, 200, 30)
    be = c3.multiselect("Engines ", [x for x in core.ENGINES if x != "exact"], ["Monte Carlo", "Scrambled Sobol", "Stratified", "Importance"])
    if st.button("Run RMSE benchmark"):
        truth, rows = core.benchmark(p, be, int(bn), int(reps))
        df = pd.DataFrame(rows).set_index("engine")
        st.markdown("Relative RMSE vs exact (lower is better). Rare-tail target is the last column.")
        st.dataframe(df.style.format("{:.3f}"))
        st.bar_chart(df[core.TARGETS[-1]])

with tabs[3]:
    st.markdown("**Oracle-call estimate: Monte Carlo vs ideal amplitude estimation** (textbook bounds, not a circuit run)")
    tg = st.selectbox("Target", core.TARGETS[1:3] + core.TARGETS[4:], index=2)
    prob = float(E[core.TARGETS.index(tg)]); rel = st.select_slider("Relative error", [.5, .2, .1, .05, .01], .1)
    if prob > 0:
        nmc, mq = core.qae_vs_mc(prob, rel)
        st.write(f"p = {prob:.2e}: Monte Carlo needs about {nmc:,.0f} samples; ideal QAE about {mq:,.0f} Grover iterations, "
                 "each a deep circuit (loader + adders + comparator). Ideal-case only; ignores noise and loader cost.")
    st.markdown("""**Braket hook.** Add an engine with the same signature `f(p, n, rng) -> dict(value, ci, queries)`
and register it in `core.ENGINES`. Estimated width for 2 units: about 40-45 qubits (3x7 loaders, 8 accumulator, ~7 loss, ~5 ancilla).""")
