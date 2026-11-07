# PowerOps Agent

I built this to explore how an AI agent could help with power-electronics validation work. You give it a bench
log from a multi-port solid-state transformer (SST) that feeds an 800 V DC bus, and it runs a validation plan
against the log, compares the run with a previous revision, checks whether the power sensors look trustworthy,
and writes up a draft report for an engineer to review.

This is a personal student project, not a product, and it isn't affiliated with any company. All the data is
synthetic and the limits in the test plan are made-up example values, not real specs. It doesn't connect to any
hardware, and it doesn't decide whether anything is ready to release. The idea came from public descriptions of
multi-port SSTs for 800 V DC data-center power (e.g. DG Matrix's Interport announcement), but nothing here is
based on company data, interfaces or specifications.

## Why

Every time a converter gets a new firmware or hardware revision, it goes through the same test sequence, and
someone has to turn a long log into a report: find each test step, compute efficiency and transient numbers,
compare them to the acceptance criteria, check whether anything got worse than the last revision, make sure a
bad number isn't just a bad sensor, and write it all up. That's slow and repetitive, so I wanted to see how much
of it an agent could draft. It stops at a draft on purpose; the report ends with an empty sign-off table.

## What it checks

The test plan lives in `config/validation_plan.example.json`. The log has a `test_step` column that says which
part of the sequence is running.

- EFF-25 / 50 / 75 / 100: steady-state efficiency (energy out / energy in) at 25, 50, 75 and 100% load.
- BURST-1: how far the 800 V bus moves on GPU-style load bursts (±80 kW, 50 ms) and how quickly it recovers.
- SAG-1: a grid sag to 0.5 pu. Looks at the lowest bus voltage and how fast the battery port picks up the load.
- THERM-1: SiC module temperature during a full-load soak, and whether it has levelled off by the end.

The plan also has regression tolerances (e.g. efficiency can't drop more than 0.2 points vs. the baseline) and
two sensor sanity checks: the battery port should read about 0 kW while the grid is healthy, and the implied
efficiency has to be physically possible.

## Demo data

The "Load demo pair" button loads two synthetic runs of the same unit. Rev-A passes all 7 tests. Rev-B fails
EFF-25, BURST-1 and THERM-1: I gave it about 0.7 kW of extra fixed loss (lower light-load efficiency and a hotter
SiC module), weaker bus control (the burst dip roughly doubles) and a slower battery response, which still
passes but shows up as a regression.

There's also a sensor-fault run (`data/SYNTHETIC_SST_revA_sensorfault.csv`) where the battery power reading has
a 3 kW offset. Every efficiency test fails, but the sensor check flags it, so the report says the failures are
probably a measurement problem rather than the hardware.

Unless you upload your own file, each page load also creates a new random unit with its own losses, control
behaviour, battery response and ambient temperature, and occasionally a hidden sensor fault. About half of them
pass everything. The random parameters are saved in the CSV header comment as an answer key but aren't given to
the agent, so it has to find problems on its own. You can turn this off in the sidebar.

## Running it (Mac)

You need Python 3.11+ (`brew install python@3.11` if you don't have it). From the `powerops-agent` folder:

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

The API key in `.env` is optional. Without it, the app runs in demo mode (no AI calls, see below). It works
with OpenAI, or with Gemini through Google's OpenAI-compatible endpoint; `.env.example` shows both.

Start the backend in one terminal:

```bash
source .venv/bin/activate && uvicorn api.main:app --reload --port 8000
```

And the UI in another:

```bash
source .venv/bin/activate && streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. To run the tests:

```bash
source .venv/bin/activate && pytest
```

## How it works

The main design rule is that the language model never calculates anything. It only decides which analysis
functions to call and with what arguments. Python computes every number and sends the results back as JSON.

1. The model gets the question, some metadata about the runs (unit ID, revision, which test steps exist), and
   descriptions of six tools. It never sees the raw samples.
2. It asks for tool calls, e.g. `run_validation_plan` or `compare_to_baseline`.
3. The backend checks the arguments against a JSON schema and runs the matching Python function. Test IDs come
   from the plan, so the model can't invent tests or change the limits.
4. The results go back to the model, which can ask for more tools (up to 8 rounds) and then writes a summary.
5. Every number in the summary is checked against the tool outputs, and anything that doesn't match is flagged.
   The tables in the report are built straight from the tool results, not from the model's text.

The tools are `run_validation_plan`, `compare_to_baseline`, `check_power_balance`, `summarize_run`,
`detect_threshold_violations` and `generate_plots`. The underlying analysis functions are in
`powerops/analysis.py`, and adding a new test type just means writing one more function there.

Demo mode skips the model completely: it picks tools with keyword rules and fills in a fixed template, so it
gives the same answer every time and needs no API key. If the API fails, the app falls back to demo mode. Requests
to control hardware (e.g. "increase the current limit") are refused, and "is this ready to release?" gets the
scorecard plus a note that release decisions are made by engineers.

## Data format

CSV, one row per sample. Lines starting with `#` are comments, and `# key: value` lines (like
`# revision: rev-B`) are read as run metadata.

Columns: `timestamp_s`, `test_step`, `grid_voltage_pu`, `grid_power_kw`, `battery_power_kw` (+ = discharging),
`rack_power_kw`, `dc_bus_voltage_v`, `sic_temp_c`.

Missing columns, empty files and malformed rows are rejected with an error message. Smaller problems (text in a
number column, unsorted or duplicate timestamps, missing steps) are fixed or flagged as warnings, and tests whose
step is missing show up as NOT RUN. `data/bad_examples/` has some broken files to try.

## Synthetic data

`powerops/synthetic.py` uses a simple model: losses are fixed + k·P², the bus has droop plus a damped ringing
response at every load step, during a grid sag the grid port only supplies 30% of demand while the battery ramps
up, and the SiC temperature follows the losses with a first-order lag (sped up so a 26-second run shows it).

```bash
python scripts/generate_synthetic.py --revision B --seed 43 --out data/my_revB.csv
```

## Project layout

- `api/main.py`: FastAPI backend
- `app/streamlit_app.py`: Streamlit UI
- `powerops/analysis.py`: the test evaluators, plan runner, baseline comparison and sensor checks
- `powerops/agent.py`: the tool-calling loop, demo mode and request filtering
- `powerops/tools.py`: tool schemas and argument checking
- `powerops/report.py`: the draft report
- `powerops/plots.py`, `synthetic.py`, `data_io.py`, `config.py`, `numbers_check.py`
- `tests/`: 112 pytest tests, including an end-to-end run through the API and a UI test

## Limitations

- Synthetic data only. Real logs would have more channels, higher sample rates and different formats.
- One run per revision, so it can't tell a small regression from normal run-to-run variation.
- The measurement methods are simplified (single-sample resolution, no uncertainty analysis).
- The number check catches made-up numbers but not a real number used in the wrong place.
- Datasets are kept in memory and there's no login, so it's meant to run locally.

## Troubleshooting

- "Cannot reach the backend": start uvicorn from the project folder.
- `ModuleNotFoundError: powerops`: run from the `powerops-agent` folder with the venv activated.
- "Fell back to demo mode": the message shows the API error. Check your key and model name.
- Port in use: run uvicorn with `--port 8001` and set `POWEROPS_API_URL=http://127.0.0.1:8001`.
