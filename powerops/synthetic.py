"""Generate SYNTHETIC validation runs for a generic multi-port SST on an 800 V DC bus.

Everything here is made up. The device model is deliberately simple:

* The run follows a fixed validation sequence (see SEQUENCE). The column
  test_step records which step is active, as a bench test sequencer would.
* Losses        = fixed_loss + k x P^2            -> efficiency curve
* Grid power    = rack power + losses (when the grid is healthy)
* 800 V bus     = nominal - droop x P, plus a damped-sine kick at every load change
* Grid sag      = grid voltage drops to 0.5 pu; the grid port can then only supply
                  30% of demand, and the battery port picks up the rest after a delay.
                  Power not yet supplied pulls the bus voltage down.
* SiC temperature follows the losses through a first-order thermal lag.
* Gaussian noise is added to every channel.

Three "revisions" make the regression story concrete:
    A = baseline (passes the illustrative plan)
    B = candidate with higher losses and weaker bus-voltage control
    C = follow-up that fixes B's bus control and most of its loss, but has a much
        slower battery-port response to a grid sag
An optional sensor fault adds an offset to the battery-port power reading.

The CSV starts with '# SYNTHETIC DATA ...' lines so it can never be mistaken
for a real measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Revision:
    label: str
    fixed_loss_kw: float  # standby + switching losses that do not scale with load
    loss_k_per_kw: float  # conduction losses = k x P^2 (P in kW)
    burst_v_per_kw: float  # size of the bus-voltage kick per kW of load change
    burst_tau_s: float  # how fast the kick decays (control-loop damping)
    battery_delay_s: float  # time before the battery port starts to respond to a sag
    battery_tau_s: float  # battery-port ramp time constant


REVISIONS = {
    "A": Revision("rev-A (baseline)", 0.8, 8e-5, 0.10, 0.003, 0.002, 0.003),
    "B": Revision("rev-B (candidate)", 1.5, 8e-5, 0.22, 0.006, 0.004, 0.004),
    "C": Revision("rev-C (candidate)", 1.0, 8e-5, 0.07, 0.0025, 0.015, 0.010),
}

# (test_step, description, duration_s, load as a fraction of rated power)
SEQUENCE = [
    (0, "warm-up", 1.0, 0.05),
    (1, "efficiency @ 25% load", 3.0, 0.25),
    (2, "efficiency @ 50% load", 3.0, 0.50),
    (3, "efficiency @ 75% load", 3.0, 0.75),
    (4, "efficiency @ 100% load", 3.0, 1.00),
    (5, "GPU load-burst test", 3.0, 0.50),
    (6, "grid-sag ride-through", 2.0, 0.75),
    (7, "thermal soak @ 100% load", 8.0, 1.00),
]
BURST_STEP, SAG_STEP = 5, 6


@dataclass
class SyntheticConfig:
    revision: str = "A"
    seed: int = 42
    unit_id: str = "SST-DEMO-001"
    sample_rate_hz: float = 1000.0
    rated_power_kw: float = 200.0
    nominal_bus_v: float = 800.0
    bus_droop_v_per_kw: float = 0.01
    burst_ring_hz: float = 80.0
    burst_extra_fraction: float = 0.4  # bursts add 40% of rated power
    burst_length_s: float = 0.05
    burst_period_s: float = 0.25
    sag_start_offset_s: float = 0.5  # seconds into the sag step
    sag_duration_s: float = 0.2
    sag_depth_pu: float = 0.5
    sag_grid_fraction: float = 0.3  # share of demand the grid port still supplies during the sag
    sag_bus_v_per_kw: float = 0.15  # bus dip per kW of demand not yet supplied
    ambient_c: float = 35.0
    thermal_r_c_per_kw: float = 20.0
    thermal_tau_s: float = 3.0  # accelerated so a short demo shows thermal behaviour
    battery_sensor_offset_kw: float = 0.0  # > 0 injects a sensor fault
    noise_kw: float = 0.15
    noise_bus_v: float = 0.3
    noise_pu: float = 0.002
    noise_c: float = 0.1
    custom_revision: Revision | None = None  # overrides `revision` (used by random_config)

    @property
    def device(self) -> Revision:
        """The device behaviour this run simulates."""
        return self.custom_revision or REVISIONS[self.revision]

    def validate(self) -> None:
        problems = []
        if self.custom_revision is None and self.revision not in REVISIONS:
            problems.append(f"revision must be one of {sorted(REVISIONS)}")
        if not 100 <= self.sample_rate_hz <= 20_000:
            problems.append("sample_rate_hz must be between 100 and 20000")
        if self.rated_power_kw <= 0 or self.nominal_bus_v <= 0:
            problems.append("rated_power_kw and nominal_bus_v must be > 0")
        if problems:
            raise ValueError("Invalid synthetic config: " + "; ".join(problems))


def generate_synthetic_run(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    """Return a DataFrame with all required columns. Deterministic for a given seed."""
    cfg = cfg or SyntheticConfig()
    cfg.validate()
    rev = cfg.device
    rng = np.random.default_rng(cfg.seed)
    fs = cfg.sample_rate_hz
    total_s = sum(d for _, _, d, _ in SEQUENCE)
    n = int(round(total_s * fs))
    t = np.arange(n) / fs

    # 1) Test sequence: step labels and commanded rack load.
    step = np.zeros(n, dtype=int)
    load = np.zeros(n)
    start = 0.0
    starts = {}
    for sid, _, dur, frac in SEQUENCE:
        mask = (t >= start - 1e-9) & (t < start + dur - 1e-9)
        step[mask], load[mask] = sid, frac * cfg.rated_power_kw
        starts[sid] = (start, start + dur)
        start += dur
    b0, b1 = starts[BURST_STEP]
    k = b0 + cfg.burst_period_s
    while k + cfg.burst_length_s <= b1 - 0.2:
        load[(t >= k - 1e-9) & (t < k + cfg.burst_length_s - 1e-9)] += cfg.burst_extra_fraction * cfg.rated_power_kw
        k += cfg.burst_period_s

    # 2) Losses and port powers with a healthy grid.
    loss = rev.fixed_loss_kw + rev.loss_k_per_kw * load**2
    demand = load + loss
    grid = demand.copy()
    battery = np.zeros(n)
    grid_pu = np.ones(n)

    # 3) Grid sag: grid port limited, battery port picks up the deficit after a delay.
    s0 = starts[SAG_STEP][0] + cfg.sag_start_offset_s
    s1 = s0 + cfg.sag_duration_s
    in_sag = (t >= s0) & (t < s1)
    grid_pu[in_sag] = cfg.sag_depth_pu
    grid[in_sag] = cfg.sag_grid_fraction * demand[in_sag]
    deficit = (1 - cfg.sag_grid_fraction) * demand
    ramp_t = t - s0 - rev.battery_delay_s
    rising = in_sag & (ramp_t > 0)
    battery[rising] = deficit[rising] * (1 - np.exp(-ramp_t[rising] / rev.battery_tau_s))
    end_value = deficit[in_sag][-1] * (1 - np.exp(-(cfg.sag_duration_s - rev.battery_delay_s) / rev.battery_tau_s))
    after = t >= s1
    battery[after] = end_value * np.exp(-(t[after] - s1) / rev.battery_tau_s)
    unsupplied = demand - grid - battery  # > 0 during the sag, < 0 just after it

    # 4) 800 V bus: droop, a kick at every load change, and the sag dip.
    bus = cfg.nominal_bus_v - cfg.bus_droop_v_per_kw * load - cfg.sag_bus_v_per_kw * unsupplied
    omega = 2 * np.pi * cfg.burst_ring_hz
    grid_t = np.linspace(0, 5 * rev.burst_tau_s, 2000)
    peak = np.max(np.exp(-grid_t / rev.burst_tau_s) * np.sin(omega * grid_t))
    for idx in np.flatnonzero(np.diff(load) != 0) + 1:
        delta_kw = load[idx] - load[idx - 1]
        tau = t[idx:] - t[idx]
        window = tau < 10 * rev.burst_tau_s
        shape = np.exp(-tau[window] / rev.burst_tau_s) * np.sin(omega * tau[window]) / peak
        bus[idx : idx + window.sum()] += -rev.burst_v_per_kw * delta_kw * shape

    # 5) SiC temperature: first-order lag toward ambient + R_th x losses (exponential moving average).
    target = cfg.ambient_c + cfg.thermal_r_c_per_kw * loss
    alpha = min(1.0, (1.0 / fs) / cfg.thermal_tau_s)
    temperature = pd.Series(target).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    df = pd.DataFrame(
        {
            "timestamp_s": np.round(t, 6),
            "test_step": step,
            "grid_voltage_pu": grid_pu + rng.normal(0, cfg.noise_pu, n),
            "grid_power_kw": grid + rng.normal(0, cfg.noise_kw, n),
            "battery_power_kw": battery + cfg.battery_sensor_offset_kw + rng.normal(0, cfg.noise_kw, n),
            "rack_power_kw": load + rng.normal(0, cfg.noise_kw, n),
            "dc_bus_voltage_v": bus + rng.normal(0, cfg.noise_bus_v, n),
            "sic_temp_c": temperature + rng.normal(0, cfg.noise_c, n),
        }
    )
    return df.round({"grid_voltage_pu": 4, "grid_power_kw": 3, "battery_power_kw": 3, "rack_power_kw": 3,
                     "dc_bus_voltage_v": 3, "sic_temp_c": 3})


def synthetic_csv_bytes(cfg: SyntheticConfig | None = None) -> bytes:
    """Generate a run and return it as CSV bytes with a SYNTHETIC banner and run metadata on top."""
    cfg = cfg or SyntheticConfig()
    df = generate_synthetic_run(cfg)
    header = [
        "# SYNTHETIC DATA - generated by the PowerOps Agent demo generator. NOT real measurements.",
        "# Generic multi-port SST model; not from any real product, company, test bench, or customer.",
        f"# unit_id: {cfg.unit_id}",
        f"# revision: {cfg.device.label}",
        "# test_sequence: " + ", ".join(f"{s}={name}" for s, name, _, _ in SEQUENCE),
        f"# generator_config: {json.dumps(asdict(cfg), separators=(',', ':'))}",
    ]
    return ("\n".join(header) + "\n" + df.to_csv(index=False)).encode("utf-8")


def random_config(seed: int | None = None) -> SyntheticConfig:
    """A run of a randomly behaving unit: new device parameters, ambient and (sometimes) a sensor fault.

    The ranges straddle the illustrative acceptance limits, so some random runs pass every test and
    others fail one or several. Pass a seed to reproduce a run; omit it for a fresh one every time.
    The drawn parameters are written into the CSV's generator_config comment (the answer key) but are
    NOT part of the run metadata the agent sees, so the agent has to find the problems itself.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))
    rng = np.random.default_rng(seed)
    device = Revision(
        label=f"random-sim #{seed}",
        fixed_loss_kw=round(float(rng.uniform(0.6, 1.4)), 3),
        loss_k_per_kw=round(float(rng.uniform(6e-5, 9.5e-5)), 7),
        burst_v_per_kw=round(float(rng.uniform(0.07, 0.2)), 3),
        burst_tau_s=round(float(rng.uniform(0.002, 0.0055)), 4),
        battery_delay_s=round(float(rng.uniform(0.001, 0.011)), 4),
        battery_tau_s=round(float(rng.uniform(0.002, 0.006)), 4),
    )
    fault = rng.random() < 0.15  # roughly 1 run in 7 has a battery power-sensor offset
    return SyntheticConfig(
        seed=seed,
        unit_id=f"SST-SIM-{seed:06d}",
        custom_revision=device,
        ambient_c=round(float(rng.uniform(25.0, 38.0)), 1),
        battery_sensor_offset_kw=round(float(rng.uniform(2.0, 5.0)), 2) if fault else 0.0,
    )
