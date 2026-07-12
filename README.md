# Antenna Surrogate Interface

An interactive **Streamlit** interface for a machine-learning **Gaussian-Process
surrogate model** of a 2.45 GHz **2×2 microstrip patch antenna array**. It is a
*forward predictor*: set 7 geometry parameters with sliders and the app shows the
model's predicted antenna performance **with uncertainty bands**, updating live.
Its distinguishing feature is **honesty** — the model predicts gain/directivity
reliably (R²=0.92) but its S11 and resonant-frequency predictions are unreliable
(R²<0) because resonance behaves discontinuously, and the UI visually communicates
exactly which predictions to trust.

## Data files (required)

Place these two provided files in the **same folder** as `app.py`:

- `surrogate_models.pkl` — pickled dict of pre-trained scikit-learn GP models,
  the fitted `StandardScaler`, and `param_cols`.
- `training_data.csv` — 54 successful full-wave CST training runs.

If the pickle fails to load (e.g. a scikit-learn version mismatch), the app
**automatically refits** the GPs from `training_data.csv`, so it stays portable.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Deploy to Streamlit Community Cloud

1. Push this folder (`app.py`, `requirements.txt`, `README.md`,
   `surrogate_models.pkl`, `training_data.csv`) to a GitHub repository.
2. Go to <https://share.streamlit.io>, click **New app**, and select the repo,
   branch, and `app.py` as the entry point.
3. Deploy. Streamlit Cloud installs `requirements.txt` automatically. If the
   installed scikit-learn version differs from the one used to train the pickle,
   the CSV-refit fallback handles it transparently.

## What the app shows

- **7 sliders** (patch L/W, inset feed depth, feed & quarter-wave widths, x/y
  element spacing), defaulting to the CST-verified champion design.
- **Four metric cards** with predicted value ± 1σ and a colour-coded trust badge
  (RELIABLE / MARGINAL / UNRELIABLE) plus each metric's R².
- An **uncertainty error-bar plot** making the contrast obvious — gain has a tiny
  bar, S11 a huge one.
- A **Predicted vs. Verified Champion** table comparing the surrogate against the
  hard-coded CST-verified truth.
- A **live 2D geometry sketch** of the 2×2 array that updates with the sliders.
- An **in-range / out-of-range** indicator tying extrapolation to growing
  uncertainty, and an "About this model" honesty note.

> The champion design was reached by manual tuning + CST verification, **not** by
> the surrogate. Any S11-critical design must be verified in full-wave simulation.
