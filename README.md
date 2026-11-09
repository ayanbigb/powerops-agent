# PowerOps Agent

An AI agent that turns power-electronics bench-test logs into draft validation reports.

You give it a CSV log from a test run of a solid-state transformer (SST) on an 800 V DC bus. It checks the run
against a test plan, compares it with an earlier hardware revision, flags possible sensor problems, and writes a
report with plots for an engineer to review.

This is a student project. All data is synthetic and the test limits are example values, not real specs.

## What it checks

- **Efficiency** at 25, 50, 75 and 100% load
- **Load bursts:** how far the 800 V bus dips during sudden load changes, and how fast it recovers
- **Grid sag:** whether the battery takes over the load quickly when grid voltage drops
- **Thermal:** peak module temperature during a full-load run
- **Regressions:** whether anything got worse compared with a previous revision
- **Sensor sanity:** whether the power readings make physical sense

The tests and limits are defined in `config/validation_plan.example.json`.

## How it works

The AI never calculates the numbers itself. It decides which checks to run, Python does the math, and the AI
writes a summary of the results. Every number in the summary is checked against the actual results, and the
report ends with a blank sign-off section for an engineer.

It works with an OpenAI or Gemini API key. Without a key, it runs in demo mode: checks are picked by keywords and
the summary is a fixed template, but all the results and plots are the same.

## Running it

Requires Python 3.11+. From the project folder:

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

Add an API key to `.env` if you have one (optional). Then start the backend:

```bash
source .venv/bin/activate && uvicorn api.main:app --port 8000
```

And in a second terminal, the app:

```bash
source .venv/bin/activate && streamlit run app/streamlit_app.py
```

Open http://localhost:8501.

## Trying it out

- **Load demo pair:** compares a good revision (rev-A, passes everything) with a worse one (rev-B, fails three
  tests).
- **New random run:** a new simulated unit each time, which may pass, fail, or have a hidden sensor fault.
  One loads automatically when you open the page.
- **Upload a CSV:** use your own log in the same format.

Then go to the **Validate with the agent** tab and ask something like "Generate the validation report" or "Did
this revision regress?"

## Data format

CSV with one row per sample and these columns: `timestamp_s`, `test_step`, `grid_voltage_pu`, `grid_power_kw`,
`battery_power_kw`, `rack_power_kw`, `dc_bus_voltage_v`, `sic_temp_c`.

Sample files are in `data/`, and some deliberately broken ones are in `data/bad_examples/`.

## Project layout

- `powerops/`: the analysis, agent, plots and report code
- `api/`: the FastAPI backend
- `app/`: the Streamlit interface
- `tests/`: automated tests (run with `pytest`)

## Limitations

- Synthetic data only; real logs would be larger and more complex.
- Simplified measurement methods.
- Meant to run locally (no login or permanent storage).
