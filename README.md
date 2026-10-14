# GridCast Live + Blocks: campus demand forecasting (Dr. N.G.P. Institute of Technology)

This is a complete, separate project folder. It already contains `web/data.json` from the earlier training run, so you can build the page straight away with `python build_page.py` (no retraining needed).

## Quick look (no setup)
Double-click `gridcast_live.html`. It opens in your browser and works offline.

## Rebuild it yourself (Windows, VS Code, Python 3.10+)
Open this folder in VS Code (File > Open Folder; you must see `train_all.py` in the Explorer), open a terminal, then run:

    python -m pip install -r requirements.txt
    python data_gen.py
    python train_all.py
    python build_page.py

Then open `gridcast_live.html`.

Fast test (about 2 minutes, lower accuracy): in PowerShell run `$env:QUICK=1; python train_all.py`, then `Remove-Item Env:QUICK` for full training again.

## What each file does
- `calendar2026.py`: official Odd Semester 2026-27 calendar (holidays, working Saturdays, tests, ESE)
- `data_gen.py`: creates the synthetic hourly campus data (`data/campus_load.csv`)
- `core.py`: features for LightGBM and Ridge
- `train_all.py`: trains LSTM, LightGBM, Ridge; writes `web/data.json`
- `build_page.py`: builds the final single-file page
- `web/`: page template, photos, Chart.js
  - `web/template.html`: now includes the diagonal photo hero, the Blocks section and the Alerts section
  - `web/img/hero.jpg`: the building photo shown in the diagonal frame (replace it to change the photo)

## Block-wise view and alerts (new)
- **Blocks:** A (MBA, EEE, ECE), B (conference hall, robotics lab), C (placement office, Texas, Cisco and language labs, placement training), D (CSE, CSBS, CSE Cyber, AIDS, IT, computer labs, halls), E (mechanical, civil) and the break zone.
- Block demand = the campus forecast of the selected model (LSTM, LightGBM, Blend or Ridge) multiplied by each block's share of the load. The share depends on the hour, the day type from the official calendar and the temperature.
- Normal, Busy and High levels for each block come from the 60th and 85th percentile of its own history.
- **Heat alert:** when the day's forecast high reaches `HOT_AT` (36 °C, set near the top of the new script block in `web/template.html`), the page writes a message asking the team to plug in the EV chargers and check the solar panels.
- **Power-cut alerts:** (1) "Test a power cut" simulates an outage and writes a generator message with the load to carry; (2) on past days, any hour where actual demand is below `DROP_AT` (72%) of the forecast is flagged as a possible power cut.
- Messages open WhatsApp (with the team number you type) or your email app, or can be copied. Automatic sending needs a server or the WhatsApp Business API.
- Block figures are estimated shares until each block has its own meter. Declare this at the event.

## Use real data
Replace `data/campus_load.csv` (columns: timestamp, load, temperature, humidity, holiday, phase) and run `train_all.py` and `build_page.py` again. Edit the dates inside `train_all.py` to match your data.

## Declare at the event
Synthetic data (including the block shares); libraries pandas, NumPy, scikit-learn, LightGBM, SHAP, TensorFlow, Chart.js.
