# INR-TSD: time-series decomposition with implicit neural representations

INR-TSD represents trend and seasonality as continuous functions using two single-hidden-layer neural networks with sinusoidal activations. The trend is fitted first; a seasonal model is then fitted to the detrended signal. Component-specific time normalization gives the networks different initial temporal scales.

This repository provides the core implementation and a nine-series synthetic demonstration associated with the INR-TSD manuscript. It contains sample inputs, example outputs, and fitting instructions. The biomedical analyses, baseline comparisons, controlled sampling-degradation experiments, and forecasting pipelines discussed in the manuscript are **not included in this package**. See [reproducibility and scope](docs/REPRODUCIBILITY.md).

This workflow does not require a user-specified seasonal period. It uses shared architectural and optimization settings, including an explicit trend initialization scale.

![Decomposition of the nine supplied synthetic series](results/decomposed_results.png)

The image and CSVs in `results/` are example outputs retained from the author-provided archive. They were not regenerated during this documentation update.

## Installation

Use **Python 3.11**. Obtain the repository by cloning its GitHub URL or extracting a release archive, then run commands from the repository root. Choose one environment method and one TensorFlow route. Dependency ranges below are installation specifications, not a frozen manuscript environment.

### 1. Create an environment

**Option A: venv (Linux/macOS/WSL2)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On native Windows, use `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell.

**Option B: conda**

```bash
conda create -n inr-tsd python=3.11 -y
conda activate inr-tsd
python -m pip install --upgrade pip
```

Conda manages Python; install the packages below with pip. Choose either venv or conda rather than nesting them.

### 2. Install CPU or GPU dependencies

**CPU: Linux x86-64 / native Windows x86-64**

```bash
python -m pip install -r requirements-cpu.txt
```

This installs `tensorflow-cpu>=2.16,<2.20` and the shared dependencies. `python -m pip install -r requirements.txt` uses this same default route.

**CPU: macOS**

```bash
python -m pip install -r requirements-macos.txt
```

This uses `tensorflow`. Intel Macs are restricted to TensorFlow 2.16.x; Apple Silicon uses the project's 2.16–2.19 range. This package does not provide a Metal GPU setup.

**NVIDIA GPU: Linux x86-64 / Windows WSL2**

```bash
nvidia-smi
python -m pip install -r requirements-gpu.txt
```

This installs `tensorflow[and-cuda]>=2.16,<2.20` and its CUDA Python dependencies. A compatible NVIDIA driver must already be installed on the host. Use separate clean environments for CPU and GPU packages. Native Windows GPU execution is not supported by this TensorFlow range; use WSL2.

Consult the official [TensorFlow installation guide](https://www.tensorflow.org/install/pip) and [version-specific build configurations](https://www.tensorflow.org/install/source#gpu) for platform and driver requirements.

### 3. Verify the installation

```bash
python -m pip check
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print(tf.reduce_sum(tf.ones((2, 2))))"
python -m unittest discover -s tests -v
```

The tensor sum should be 4. A CPU installation may report `GPU: []`; a GPU installation should list a device. Tests check widths, initialization, and a short fit, not manuscript-level reproduction. Fresh CPU/GPU installations and TensorFlow training were not run during this packaging update.

## Run the synthetic demonstration

```bash
python main.py
```

The script fits nine series, writes component CSVs and `results/decomposed_results.{png,eps}`, and displays a figure. It creates `model/` and `results/` if needed. On a headless Linux server:

```bash
MPLBACKEND=Agg PYTHONHASHSEED=2 TF_DETERMINISTIC_OPS=1 python main.py
```

Inputs are supplied. To rebuild them:

```bash
MPLBACKEND=Agg python data_generation.py
```

Keep `dataset/trend3.csv`: its third column is the reference for trend type 3. Regeneration overwrites the nine input CSVs and input figures. Running `main.py` overwrites example output and checkpoint files; preserve a copy when comparing runs.

Each input has `time`, `value`, `trend`, and `seasonality` columns and 100 observations on [0, 50]. Training uses only the first two columns. Output files are named `results/season{s}_trend{t}_decomposed.csv`:

| Column | Meaning |
| --- | --- |
| `time` | Observation time |
| `actual_value` | Observed signal |
| `predicted_trend` | Estimated trend in original units |
| `predicted_seasonality` | Estimated seasonality in original units |

The remainder is `actual_value - predicted_trend - predicted_seasonality`. This is in-sample decomposition, not a held-out forecasting evaluation. Verified installation and demo timings were not supplied; measure them on the release environment.

## Configuration

Edit `config.py` **before starting a new Python process**; imported settings are not a live configuration interface.

### Observation-dependent seasonal width

The implementation uses $H_T=40$ and $H_S=\lceil N/2.5\rceil$, where `N = len(x_season)` is the number of observations in one input series, not batch size or number of series.

| N | Trend nodes | Seasonal nodes |
| --- | --- | --- |
| 100 | 40 | 40 |
| 101 | 40 | 41 |
| 240 | 40 | 96 |
| 250 | 40 | 100 |

`TREND_HIDDEN_UNITS` controls trend width; `SEASONAL_WIDTH_DIVISOR = 2.5` controls the empirical seasonal rule. Initialization reads each layer's actual width.

The manuscript's controlled sampling-degradation comparisons held widths fixed using the undegraded series. This release recalculates seasonal width for each input and does not implement that experimental override. See [scope and limitations](docs/REPRODUCIBILITY.md).

### Set the trend initialization scale

Trend time is normalized to $\tau=(t-a)/(b-a)\in[0,1]$. A term $\sin(w\tau+\phi)$ completes $|w|/(2\pi)$ cycles over this window. Initial frequency weights are equally spaced in $[0,\Omega_T]$, where

$$\Omega_T=2\pi C_T^0.$$

Set $C_T^0$ through `TREND_INIT_MAX_CYCLES` in `config.py`:

```python
TREND_HIDDEN_UNITS = 40
TREND_INIT_MAX_CYCLES = 1 / (2 * math.pi)
TREND_INIT_MAX_FREQUENCY = 2 * math.pi * TREND_INIT_MAX_CYCLES
```

The default gives [0, 1], or approximately 0.159 initial cycles. For **one initial cycle over the entire observation window**, change only:

```python
TREND_INIT_MAX_CYCLES = 1.0
```

Keep the derived `TREND_INIT_MAX_FREQUENCY` expression and restart `python main.py`. A cycle value of `1.0` gives a frequency upper bound of $2\pi$, not 1.

| Initial frequency range | Set `TREND_INIT_MAX_CYCLES` to | Maximum initial cycles |
| --- | --- | --- |
| [0, 1], default | `1 / (2 * math.pi)` | 0.159 |
| [0, 2π] | `1.0` | 1.000 |
| [0, 10] | `10 / (2 * math.pi)` | 1.592 |
| [0, 50] | `50 / (2 * math.pi)` | 7.958 |

`fit_trend()` passes the derived upper bound to `seed_first_layer_frequencies()`, which assigns `np.linspace(0.0, max_frequency, layer.units)` to the first-layer kernel. This overwrites the builder's temporary constant initializer. Seasonality separately uses `SEASONAL_INIT_MAX_FREQUENCY = 1.0`.

Frequencies remain trainable: this is an initialization scale, not a hard cutoff. Larger ranges allow faster variation in the initial trend basis and may assign oscillatory structure to the trend. Choose and report the scale in relation to the scientific question. The manuscript default holds [0, 1] fixed across datasets; alternatives are different configurations.

### Training defaults

| Setting | Default | Meaning |
| --- | --- | --- |
| `INITIAL_LR` | `1e-7` | Initial learning rate |
| `TREND_TARGET_LR` | `0.1` | Trend target learning rate |
| `SEASONALITY_TARGET_LR` | `0.01` | Seasonal target learning rate |
| `WARMUP_STEPS` | `100` | Warmup optimizer steps |
| `DECAY_STEPS` | `200` | First zero of the post-warmup cosine |
| `BATCH_SIZE` | `500` | Single full batch for N=100 |
| `TREND_EPOCHS` | `500` | Trend epochs |
| `SEASONALITY_EPOCHS` | `(500, 200, 200)` | Seasonal stages; fresh optimizer per stage |
| `EARLY_STOPPING_PATIENCE` | `1000` | Loss-monitoring patience |
| `GLOBAL_TF_SEED` / `DATA_SEED` | `2` / `41` | Configured RNG seeds |

Both models use Adam and mean absolute error. The current cosine is **not clamped after step 200** and can rise again. Steps are optimizer updates, not epochs; larger inputs may require several steps per epoch. See [implementation notes](docs/REPRODUCIBILITY.md).

## Use another time series

```bash
python examples/decompose_csv.py --input path/to/series.csv --output results/custom_decomposed.csv
```

The example requires `time` and `value` columns, finite numeric values, strictly increasing timestamps, at least three observations, and nonconstant values. It shifts time to start at zero for fitting, retains original timestamps in the output, and creates checkpoint directories. The origin shift accounts for the current seasonal scaling `x * pi * N / x.max()`. Missing values are rejected rather than imputed. The core function itself does not perform these checks.

## Repository contents

```text
config.py                  Model, initialization, training and seed settings
main.py                    Nine-series demonstration
data_generation.py         Input generation
requirements*.txt          Shared and platform-specific dependencies
source/                    Models, normalization, training and generators
dataset/                   Nine inputs, reference trend and input figures
results/                   Supplied example estimates and figures
examples/decompose_csv.py   Custom-CSV example
tests/test_models.py       Model and initialization checks
docs/REPRODUCIBILITY.md    Scope, implementation and validation record
docs/RELEASE_CHECKLIST.md  Author tasks before public release
CHANGELOG.md               Changes relative to the supplied archive
LICENSE                    Supplied MIT license
```

Transient weights are generated in `model/` and excluded from the source release. Fixed checkpoint names are reused across series; they are not a per-series model archive. Supplied input and result CSVs remain tracked.

## Reproducibility and code availability

Record exact package versions, OS, CPU/GPU, settings, runtime, and source commit for the manuscript run. Seeds alone do not establish bitwise equivalence across versions or devices. See [reproducibility](docs/REPRODUCIBILITY.md) and the [release checklist](docs/RELEASE_CHECKLIST.md).

Final GitHub URL, release identifier, persistent archive DOI, and manuscript citation must be supplied by the authors before submission. The checklist includes a code-availability template and identifies experiments absent from this package. Publication or acceptance by Nature Communications is not asserted.

## Citation and license

Cite the manuscript and the exact archived software release used. Final bibliographic metadata and a software DOI were not supplied; add verified metadata when available. The previous README listed Gyuyoung Hwang, Jinwoo Hyun, and Jae Kyoung Kim; verify author order and final title before release.

The supplied [MIT License](LICENSE) is retained. Its copyright-holder placeholder must be completed by the rights holder. After publication, use the repository's Issues tab for reproducible problems, including versions, settings, and a shareable minimal example.
