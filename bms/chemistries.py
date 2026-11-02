"""
Battery chemistry profiles: OCV-SOC lookup tables and cell parameter presets.

Established chemistries
-----------------------
LFP        — LiFePO4: flat OCV plateau, long cycle life, safe       (EKF critical)
PbA        — Lead-Acid: mature, cheap, high self-discharge           (retrofit EVs)
NMC        — Li(NiMnCo)O2: high energy density, cars & buses        (steep OCV)

Emerging / next-generation chemistries
---------------------------------------
NA_ION     — Sodium-Ion: no lithium, excellent cold performance, entering mass
              production (CATL 2023+, HiNa). Slightly lower energy than LFP.
LMFP       — Lithium Manganese Iron Phosphate: phosphate frame with Mn giving
              ~20 % more energy than LFP; dual Fe/Mn redox plateau; BYD Blade II.
LTO        — Lithium Titanate Oxide anode (Li4Ti5O12): extreme fast-charge (10C+),
              100 k-cycle life, −40 °C operation; low energy density; grid/transit.
LI_SULFUR  — Lithium-Sulfur: highest theoretical energy (2600 Wh/kg), flat 2.1 V
              plateau; polysulfide shuttle limits cycle life to ~500 (improving).
              Aerospace (Airbus, 2024 flight demo), future EV packs.
SOLID_STATE — Solid-State NMC: solid oxide / sulfide electrolyte eliminates liquid,
              widens voltage window to 4.3 V, suppresses dendrites. Toyota targets
              2027-28 production. Current constraint: needs warm operating temp.

Usage
-----
    from bms.chemistries import LFP, LEAD_ACID, NMC, NA_ION, LMFP, LTO
    cell = CellModel(config=LFP.cell_config, ocv_table=LFP.ocv_table)
"""

from dataclasses import dataclass

import numpy as np

from .config import CellConfig


@dataclass(frozen=True)
class ChemistryProfile:
    name: str
    symbol: str
    ocv_table: np.ndarray           # [SOC, OCV_per_cell]
    cell_config: CellConfig
    self_discharge_pct_month: float # %/month
    charge_efficiency: float        # round-trip fraction (0–1)
    float_voltage_v: float          # Stage-3 float charge voltage per cell
    description: str


# ---------------------------------------------------------------------------
# LiFePO4 (LFP) — used in modern 2-wheelers, 4-wheelers, buses
# ---------------------------------------------------------------------------
_LFP_OCV = np.array([
    [0.00, 2.500],
    [0.05, 3.000],
    [0.10, 3.120],
    [0.20, 3.180],
    [0.30, 3.210],
    [0.40, 3.230],
    [0.50, 3.240],
    [0.60, 3.250],
    [0.70, 3.260],
    [0.80, 3.280],
    [0.90, 3.320],
    [0.95, 3.400],
    [1.00, 3.650],
])

_LFP_CELL = CellConfig(
    nominal_capacity_ah=100.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=3.20, v_max=3.65, v_min=2.50,
    r0=0.002, r1=0.001, c1=3000.0, r2=0.002, c2=30000.0,
    i_max_charge=-50.0, i_max_discharge=200.0,
    i_cc_charge=-50.0, i_cv_term=-5.0,
    t_max_charge=45.0, t_min_charge=0.0,
    t_max_discharge=60.0, t_min_discharge=-20.0,
    t_nominal=25.0,
    thermal_capacity_j_k=1500.0, thermal_resistance_k_w=5.0,
)

LFP = ChemistryProfile(
    name="Lithium Iron Phosphate",
    symbol="LFP",
    ocv_table=_LFP_OCV,
    cell_config=_LFP_CELL,
    self_discharge_pct_month=2.0,
    charge_efficiency=0.97,
    float_voltage_v=3.40,
    description="Safe, long-cycle-life LFP. Flat OCV needs EKF for accurate SOC.",
)


# ---------------------------------------------------------------------------
# Lead-Acid (PbA) — used in retrofit EVs (IJAREEIE paper: 72V, 130Ah pack)
#
# Pack from paper: 6 × 12 V modules = 72 V, each module = 6 cells × 2 V
# Total cells: 36 cells × 2 V = 72 V.
# Charging: 3-stage (CC bulk → CV absorption → Float)
# Optimum operating: 25–33 °C; dangerous above 50 °C
# ---------------------------------------------------------------------------
_PBA_OCV = np.array([
    [0.00, 1.750],   # severely over-discharged
    [0.10, 1.870],
    [0.20, 1.920],
    [0.30, 1.960],
    [0.40, 1.980],
    [0.50, 2.000],   # nominal
    [0.60, 2.020],
    [0.70, 2.060],
    [0.80, 2.100],
    [0.90, 2.150],
    [0.95, 2.200],
    [1.00, 2.300],   # open-circuit fully charged
])

_PBA_CELL = CellConfig(
    nominal_capacity_ah=130.0,          # per IJAREEIE paper
    soc_min=0.20, soc_max=0.95,         # lead-acid shouldn't go below 20 %
    v_nominal=2.00, v_max=2.40, v_min=1.75,
    # Internal resistance for a large 130 Ah VRLA cell is low (~1-2 mΩ).
    # Values scale inversely with capacity: small cells have higher R.
    r0=0.001, r1=0.001, c1=5000.0,
    r2=0.001, c2=100000.0,
    i_max_charge=-10.0,                 # charger in paper: 0 to 10 A
    i_max_discharge=130.0,              # 1 C
    i_cc_charge=-10.0,                  # CC charge current
    i_cv_term=-1.0,                     # ~0.01 C float threshold
    t_max_charge=40.0,                  # paper: below 40 °C
    t_min_charge=10.0,                  # paper: 10–50 °C range
    t_max_discharge=50.0,               # paper: 50 °C is dangerous
    t_min_discharge=0.0,
    t_nominal=29.0,                     # paper: 25–33 °C optimum, mid = 29
    thermal_capacity_j_k=4000.0,        # lead-acid is thermally massive
    thermal_resistance_k_w=3.0,
)

LEAD_ACID = ChemistryProfile(
    name="Lead-Acid (VRLA)",
    symbol="PbA",
    ocv_table=_PBA_OCV,
    cell_config=_PBA_CELL,
    self_discharge_pct_month=10.0,      # 3–20 %, mid-range per paper
    charge_efficiency=0.72,             # 50–95 %, conservative
    float_voltage_v=2.27,               # standard float ~2.25–2.30 V/cell
    description=(
        "Lead-acid 2V cells. 3-stage CC→Topping→Float charging. "
        "Retrofit EV pack from IJAREEIE paper: 36S × 2V = 72V, 130Ah."
    ),
)


# ---------------------------------------------------------------------------
# NMC (Li(NiMnCo)O2) — higher energy density, used in premium 4-wheelers
# ---------------------------------------------------------------------------
_NMC_OCV = np.array([
    [0.00, 3.000],
    [0.05, 3.420],
    [0.10, 3.500],
    [0.20, 3.620],
    [0.30, 3.680],
    [0.40, 3.730],
    [0.50, 3.780],
    [0.60, 3.840],
    [0.70, 3.920],
    [0.80, 4.010],
    [0.90, 4.100],
    [0.95, 4.160],
    [1.00, 4.200],
])

_NMC_CELL = CellConfig(
    nominal_capacity_ah=60.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=3.70, v_max=4.20, v_min=3.00,
    r0=0.002, r1=0.001, c1=2000.0, r2=0.003, c2=20000.0,
    i_max_charge=-60.0,                 # 1 C
    i_max_discharge=300.0,              # 5 C peak
    i_cc_charge=-60.0,
    i_cv_term=-3.0,                     # 0.05 C
    t_max_charge=45.0, t_min_charge=0.0,
    t_max_discharge=60.0, t_min_discharge=-20.0,
    t_nominal=25.0,
    thermal_capacity_j_k=1200.0, thermal_resistance_k_w=4.0,
)

NMC = ChemistryProfile(
    name="Lithium Nickel Manganese Cobalt Oxide",
    symbol="NMC",
    ocv_table=_NMC_OCV,
    cell_config=_NMC_CELL,
    self_discharge_pct_month=1.5,
    charge_efficiency=0.98,
    float_voltage_v=3.90,
    description="High energy density NMC. Steep OCV aids SOC estimation. Premium EVs.",
)

# ---------------------------------------------------------------------------
# Sodium-Ion (Na-Ion)  —  hard-carbon anode / layered-oxide cathode
#
# Commercial status: CATL first batch 2023; entering EV production 2024-25.
# Key advantages over LFP:
#   - No lithium, cobalt or nickel in anode → lower material cost
#   - Excellent low-temperature performance (discharge possible at −30 °C)
#   - Flat OCV at ~3.1 V; cell voltage range 2.0–4.0 V
# Key limitation: ~15 % lower gravimetric energy density than LFP.
# OCV shape: sloped rather than flat (easier SOC estimation than LFP).
# ---------------------------------------------------------------------------
_NAION_OCV = np.array([
    [0.00, 2.000],   # completely discharged
    [0.05, 2.500],
    [0.10, 2.800],
    [0.20, 3.000],
    [0.30, 3.100],
    [0.40, 3.150],
    [0.50, 3.200],   # mid-SOC plateau (~3.1–3.2 V)
    [0.60, 3.250],
    [0.70, 3.300],
    [0.80, 3.400],
    [0.90, 3.550],
    [0.95, 3.650],
    [1.00, 4.000],
])

_NAION_CELL = CellConfig(
    nominal_capacity_ah=100.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=3.10, v_max=4.00, v_min=2.00,
    r0=0.004, r1=0.002, c1=2500.0, r2=0.003, c2=25000.0,
    i_max_charge=-100.0,            # 1 C charge (conservative for first-gen)
    i_max_discharge=300.0,          # 3 C continuous
    i_cc_charge=-100.0,
    i_cv_term=-5.0,
    t_max_charge=45.0,
    t_min_charge=-20.0,             # excellent cold-charge capability vs Li-ion
    t_max_discharge=60.0,
    t_min_discharge=-30.0,          # operates at −30 °C (LFP: −20 °C)
    t_nominal=25.0,
    thermal_capacity_j_k=1400.0,
    thermal_resistance_k_w=5.0,
)

NA_ION = ChemistryProfile(
    name="Sodium-Ion (Na-Ion)",
    symbol="Na-Ion",
    ocv_table=_NAION_OCV,
    cell_config=_NAION_CELL,
    self_discharge_pct_month=3.0,
    charge_efficiency=0.96,
    float_voltage_v=3.50,
    description=(
        "Hard-carbon / layered-oxide Na-Ion. No Li/Co/Ni in anode. "
        "~15 % lower energy than LFP but better cold performance and lower cost. "
        "CATL first commercial batch 2023; EV packs 2024+."
    ),
)


# ---------------------------------------------------------------------------
# LMFP — Lithium Manganese Iron Phosphate  (LiMn₀.₆Fe₀.₄PO₄)
#
# Commercial status: BYD 'Blade' generation-2, CATL M3P (2023+).
# Advantages over LFP:
#   - ~20 % higher energy density (3.6 V vs 3.2 V nominal)
#   - Same olivine safety profile as LFP (no thermal runaway risk)
#   - Dual redox plateau: Fe²⁺/³⁺ at ~3.50 V, Mn²⁺/³⁺ at ~3.70 V
# Limitation: slightly lower cycle life than pure LFP; Mn dissolution at
#   high temperatures reduces longevity above 45 °C.
# OCV: two-step plateau (flat with a visible kink at ~50 % SOC).
# ---------------------------------------------------------------------------
_LMFP_OCV = np.array([
    [0.00, 2.500],
    [0.05, 2.900],
    [0.10, 3.200],
    [0.20, 3.420],
    [0.30, 3.500],   # Fe²⁺/³⁺ lower plateau begins
    [0.40, 3.520],
    [0.50, 3.540],   # kink: Fe→Mn plateau transition
    [0.60, 3.660],   # Mn²⁺/³⁺ upper plateau begins
    [0.70, 3.680],
    [0.80, 3.720],
    [0.90, 3.750],
    [0.95, 3.770],
    [1.00, 3.800],
])

_LMFP_CELL = CellConfig(
    nominal_capacity_ah=80.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=3.60, v_max=3.80, v_min=2.50,
    r0=0.003, r1=0.001, c1=2800.0, r2=0.002, c2=28000.0,
    i_max_charge=-80.0,             # 1 C
    i_max_discharge=320.0,          # 4 C continuous
    i_cc_charge=-80.0,
    i_cv_term=-4.0,
    t_max_charge=45.0,              # Mn dissolution accelerates above 45 °C
    t_min_charge=0.0,
    t_max_discharge=60.0,
    t_min_discharge=-20.0,
    t_nominal=25.0,
    thermal_capacity_j_k=1300.0,
    thermal_resistance_k_w=4.5,
)

LMFP = ChemistryProfile(
    name="Lithium Manganese Iron Phosphate",
    symbol="LMFP",
    ocv_table=_LMFP_OCV,
    cell_config=_LMFP_CELL,
    self_discharge_pct_month=2.0,
    charge_efficiency=0.97,
    float_voltage_v=3.60,
    description=(
        "LiMn₀.₆Fe₀.₄PO₄ olivine. ~20 % more energy than LFP; same safety. "
        "Dual plateau (Fe & Mn). BYD Blade Gen-2, CATL M3P (2023+). "
        "Avoid sustained cycling above 45 °C to limit Mn dissolution."
    ),
)


# ---------------------------------------------------------------------------
# LTO — Lithium Titanate Oxide  (Li₄Ti₅O₁₂ anode / LMO or NMC cathode)
#
# Commercial status: mature, used in city buses (Proterra, Yinlong), grid
#   storage, and fast-charge transit (Toshiba SCiB since 2008).
# Advantages:
#   - 10–20 C fast-charge without lithium plating (no graphite)
#   - 10 000–30 000 cycle life (spinel structure survives volume change)
#   - Operation from −40 °C to +65 °C
#   - Zero-strain intercalation → excellent safety
# Limitations: cell voltage only 2.3 V (half of LFP); low energy density.
# OCV: extremely flat plateau at ~2.33 V (great for SOC but needs high-res ADC).
# ---------------------------------------------------------------------------
_LTO_OCV = np.array([
    [0.00, 1.500],
    [0.05, 2.000],
    [0.10, 2.180],
    [0.20, 2.280],
    [0.30, 2.310],
    [0.40, 2.320],
    [0.50, 2.330],   # ultra-flat LTO plateau
    [0.60, 2.345],
    [0.70, 2.360],
    [0.80, 2.420],
    [0.90, 2.530],
    [0.95, 2.610],
    [1.00, 2.700],
])

_LTO_CELL = CellConfig(
    nominal_capacity_ah=30.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=2.30, v_max=2.70, v_min=1.50,
    r0=0.003, r1=0.001, c1=2000.0, r2=0.001, c2=15000.0,
    i_max_charge=-300.0,            # 10 C (fast-charge hallmark of LTO)
    i_max_discharge=450.0,          # 15 C peak
    i_cc_charge=-300.0,
    i_cv_term=-3.0,
    t_max_charge=55.0,              # safe at elevated temp (zero-strain anode)
    t_min_charge=-30.0,             # excellent cold charge vs graphite
    t_max_discharge=65.0,
    t_min_discharge=-40.0,          # widest operating range of any Li chemistry
    t_nominal=25.0,
    thermal_capacity_j_k=900.0,     # smaller cells → lower thermal mass
    thermal_resistance_k_w=6.0,
)

LTO = ChemistryProfile(
    name="Lithium Titanate Oxide",
    symbol="LTO",
    ocv_table=_LTO_OCV,
    cell_config=_LTO_CELL,
    self_discharge_pct_month=1.0,   # lower self-discharge than NMC
    charge_efficiency=0.99,         # near-unity (low polarisation)
    float_voltage_v=2.40,
    description=(
        "Li₄Ti₅O₁₂ anode. Extreme fast-charge (10 C+), 30 k-cycle life, "
        "−40 °C to +65 °C. Low 2.3 V cell voltage → low energy density. "
        "City buses (Proterra, Yinlong), grid storage, Toshiba SCiB."
    ),
)


# ---------------------------------------------------------------------------
# Li-S — Lithium-Sulfur
#
# Commercial status: early aerospace deployment (OXIS, Sion Power, Lyten).
#   Airbus UpNext flew a Li-S pack in A350 trials 2024. EV versions targeted
#   for 2026-28 (Lyten, 3x current Li-ion range claim).
# Advantages:
#   - Highest theoretical energy: 2600 Wh/kg (practical ~400–550 Wh/kg today)
#   - Low cost (sulfur is abundant)
#   - Performance improves at high temperature (unlike Li-ion)
# Limitations:
#   - Polysulfide shuttle → capacity fade; cycle life ~300–800 (improving fast)
#   - Volume expansion of S cathode
#   - High self-discharge (~10 %/month)
#   - Flat 2.1 V OCV → EKF must rely on coulomb counting (similar to LFP)
# OCV: two plateaus — upper at ~2.4 V (S8→Li₂S₄) and lower at ~2.1 V (→Li₂S).
# ---------------------------------------------------------------------------
_LIS_OCV = np.array([
    [0.00, 1.500],
    [0.05, 1.800],
    [0.10, 2.050],
    [0.20, 2.100],   # lower plateau (Li₂S₄ → Li₂S)
    [0.40, 2.120],
    [0.50, 2.150],   # transition valley between plateaus
    [0.55, 2.300],
    [0.60, 2.380],   # upper plateau (S8 → Li₂S₄)
    [0.70, 2.400],
    [0.80, 2.420],
    [0.90, 2.450],
    [0.95, 2.550],
    [1.00, 2.800],
])

_LIS_CELL = CellConfig(
    nominal_capacity_ah=20.0,       # conservative practical for first-gen cells
    soc_min=0.10, soc_max=0.90,     # avoid deep discharge (Li₂S blocking layer)
    v_nominal=2.10, v_max=2.80, v_min=1.50,
    r0=0.010, r1=0.005, c1=3000.0, r2=0.008, c2=40000.0,
    i_max_charge=-20.0,             # 1 C max (slow charge preferred)
    i_max_discharge=60.0,           # 3 C peak; sustained at 1 C
    i_cc_charge=-20.0,
    i_cv_term=-2.0,
    t_max_charge=45.0,
    t_min_charge=0.0,               # cold performance limited vs LTO
    t_max_discharge=60.0,
    t_min_discharge=-10.0,
    t_nominal=30.0,                 # benefits from slightly elevated temperature
    thermal_capacity_j_k=800.0,
    thermal_resistance_k_w=7.0,
)

LI_SULFUR = ChemistryProfile(
    name="Lithium-Sulfur",
    symbol="Li-S",
    ocv_table=_LIS_OCV,
    cell_config=_LIS_CELL,
    self_discharge_pct_month=10.0,  # polysulfide shuttle drives high self-discharge
    charge_efficiency=0.85,         # lower round-trip due to shuttle losses
    float_voltage_v=2.30,
    description=(
        "Sulfur cathode / Li metal anode. Theoretical 2600 Wh/kg; practical "
        "~400-550 Wh/kg today. Two-plateau OCV (2.1 V + 2.4 V). "
        "Cycle life limited (~500 cycles, improving). Airbus A350 flight demo "
        "2024; Lyten targeting EV packs 2026-28."
    ),
)


# ---------------------------------------------------------------------------
# Solid-State NMC — NMC811 with solid oxide/sulfide electrolyte
#
# Commercial status: pre-production / advanced prototype (Toyota, QuantumScape,
#   Solid Power, Samsung SDI). Toyota targets 2027-28 series EV production.
# Advantages:
#   - Solid electrolyte eliminates liquid → no thermal runaway
#   - Wider voltage window (up to 4.3 V vs 4.2 V liquid Li-ion)
#   - Suppresses Li dendrites → enables Li metal or Si anode (future)
#   - Higher energy density than conventional NMC
# Current limitations:
#   - Solid electrolyte has lower ionic conductivity at room temperature →
#     cells need to be pre-heated above ~30 °C for adequate rate capability
#   - High manufacturing cost; yield challenges at scale
#   - t_min_charge: 30 °C (solid electrolyte constraint)
# OCV: similar to NMC811 but slightly wider window (to 4.3 V).
# ---------------------------------------------------------------------------
_SSLFP_OCV = np.array([
    [0.00, 2.700],
    [0.05, 3.300],
    [0.10, 3.550],
    [0.20, 3.670],
    [0.30, 3.720],
    [0.40, 3.770],
    [0.50, 3.830],
    [0.60, 3.920],
    [0.70, 4.020],
    [0.80, 4.100],
    [0.90, 4.180],
    [0.95, 4.240],
    [1.00, 4.300],
])

_SSLFP_CELL = CellConfig(
    nominal_capacity_ah=60.0,
    soc_min=0.05, soc_max=0.95,
    v_nominal=3.80, v_max=4.30, v_min=2.70,
    r0=0.001, r1=0.001, c1=2000.0, r2=0.002, c2=20000.0,
    i_max_charge=-60.0,             # 1 C (conservative for pre-production)
    i_max_discharge=300.0,          # 5 C peak
    i_cc_charge=-60.0,
    i_cv_term=-3.0,
    t_max_charge=60.0,
    t_min_charge=30.0,              # solid electrolyte requires pre-heat
    t_max_discharge=80.0,           # wider upper limit (no liquid to boil/leak)
    t_min_discharge=0.0,            # cold performance limited by ionic conductivity
    t_nominal=40.0,                 # optimal operating: 40–55 °C
    thermal_capacity_j_k=1100.0,
    thermal_resistance_k_w=4.0,
)

SOLID_STATE = ChemistryProfile(
    name="Solid-State NMC811",
    symbol="SS-NMC",
    ocv_table=_SSLFP_OCV,
    cell_config=_SSLFP_CELL,
    self_discharge_pct_month=0.5,   # very low (no liquid electrolyte leakage paths)
    charge_efficiency=0.99,
    float_voltage_v=4.00,
    description=(
        "NMC811 cathode with solid oxide/sulfide electrolyte. No thermal runaway; "
        "4.3 V upper voltage; higher energy density than liquid Li-ion. "
        "Requires >30 °C for adequate rate capability. Toyota / QuantumScape / "
        "Solid Power targeting 2027-28 production."
    ),
)


# Registry for easy lookup
CHEMISTRY_REGISTRY = {
    # Established
    "LFP":        LFP,
    "PbA":        LEAD_ACID,
    "NMC":        NMC,
    # Emerging / next-generation
    "Na-Ion":     NA_ION,
    "LMFP":       LMFP,
    "LTO":        LTO,
    "Li-S":       LI_SULFUR,
    "SS-NMC":     SOLID_STATE,
}
