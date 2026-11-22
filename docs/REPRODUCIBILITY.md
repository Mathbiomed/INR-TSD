# Reproducibility and implementation record

## Scope of this distribution

This package was prepared from `new_INR_TSD.zip` supplied by the authors. It implements sequential trend/seasonality fitting and a 3-by-3 synthetic demonstration. Source configuration values and numerical training operations have been preserved; documentation, dependency routes, a custom-data example, and tests have been added.

| Workflow | Included? | Entry point or limitation |
| --- | --- | --- |
| Generate nine synthetic inputs | Yes | `data_generation.py`; requires supplied `dataset/trend3.csv` |
| Fit nine synthetic inputs | Yes | `main.py` |
| Fit one custom CSV | Yes | `examples/decompose_csv.py` |
| Ground-truth components | Yes | Input `trend` and `seasonality` columns |
| Supplied example outputs | Yes | `results/`; inherited, not regenerated here |
| Biomedical datasets and analyses | No | Provide data-access and analysis instructions separately |
| Baseline methods and paper metric tables | No | Add corresponding scripts and figure/table mapping |
| Forecasting pipeline | No | Separate implementation needed |
| Controlled sampling degradation | No | Requires fixed-width protocol described below |

## Input and fitting details

The provided generator uses 100 equally spaced times on [0, 50]. Seasonal signals are `sin(2*x)`, `sin(x) + sin(2*x)`, and `sin(0.5*x) + sin(sqrt(2)*x)`. Trends are `0.2*x`, `exp(-0.05*x + 2)`, and the third column of `dataset/trend3.csv`. The archive does not specify how that third trend was originally produced; it is retained as a required reference input.

Trend inputs are min-max normalized to [0, 1]; targets are normalized to [-1, 1]. The seasonal model fits the normalized signal minus the fitted trend. Components are then returned in original value units. Zero target range and zero time range are undefined in these normalization functions. The custom example rejects such inputs and nonfinite values. It does not impute missing measurements.

Current seasonal scaling is `x * pi * N / x.max()`. This matches `pi * N * (t-a)/(b-a)` for zero-origin inputs. The custom example shifts time to zero before calling the core routine and restores the original timestamps in the output. Do not assume the core function itself is invariant to arbitrary timestamp origins.

The default widths are trend 40 and seasonal `ceil(N/2.5)`. Both first layers are initialized with evenly spaced weights over their configured frequency intervals. They remain trainable. Omitting the seasonal output intercept does not mathematically guarantee a zero empirical mean of the seasonal prediction.

For the revised manuscript's controlled sampling-degradation experiments, network widths were determined from the undegraded input and held fixed across its degraded versions. This package's `fit_seasonality()` recomputes width from the current input length and has no reference-count override. Do not label a run with fewer observations as a reproduction of that fixed-width experiment without implementing and recording the protocol.

## Optimization and checkpoint behavior

Both components use Adam and MAE. The scheduler operates on optimizer updates. At N=100 and batch size 500, one epoch is one update. Seasonal stages recompile the model with a new optimizer, restarting its schedule while retaining learned weights.

`WarmUpCosineDecay` performs a linear warmup to step 100, then evaluates an unclamped cosine. Its first zero is step 200; it rises again afterwards. This behavior is preserved from the supplied implementation. Verify that the scheduler used for the manuscript results is the same; changing it requires rerunning and checking the affected results.

Callbacks monitor training loss, not validation loss. Seasonal checkpoint callbacks are recreated per stage, and the final load retrieves the best checkpoint of the final stage, not a guaranteed best checkpoint across all three stages. Early stopping patience is 1000, exceeding each stage's epoch count.

`model/trend.weights.h5` and `model/season.weights.h5` are reused for each series. The weights do not carry per-series configuration metadata. They are omitted from this source package because the supplied pair does not document which series and settings produced it. Avoid concurrent runs sharing these paths. Save separate checkpoints and settings if per-series inference artifacts are required.

## Randomness and environment

The code sets TensorFlow and NumPy seeds in `configure_environment()`, then resets Python and NumPy seeds before each series. It does not reset TensorFlow for every series, so fitting a series alone can differ from its fit within the nine-series loop. Existing random-state handling has been retained.

Set `PYTHONHASHSEED` and `TF_DETERMINISTIC_OPS` before launching Python as shown in the README. Setting `PYTHONHASHSEED` after interpreter startup does not retroactively fix hash randomization. Seed settings alone are not evidence of bitwise determinism across hardware or package versions.

The dependency specifications retain the supplied broad version ranges; they are not a tested version matrix or a lockfile. After a successful reference run, record the actual environment:

```bash
python --version
python -m pip freeze > environment-lock.txt
python -c "import platform, tensorflow as tf; print(platform.platform()); print(tf.__version__); print(tf.config.list_physical_devices()); print(tf.sysconfig.get_build_info())"
git rev-parse HEAD
```

For GPU runs also record `nvidia-smi`. Save the settings, hardware, input provenance, run command, installation time, and elapsed demo time alongside the release. Review machine-specific paths in exported environment files before publishing them.

## Verification status

During packaging, Python syntax, documentation links, configuration arithmetic, initialization logic in isolation, and supplied CSV consistency were checked. Input and example-result files were retained unchanged from the supplied archive. TensorFlow was unavailable in the packaging environment, so full model execution, the TensorFlow tests, and CPU/GPU installation checks were not run. The supplied example results must not be described as newly reproduced.

Run `python -m unittest discover -s tests -v` after installing TensorFlow. These tests check all seasonal nodes receive the requested frequency range, including widths below and above 40; verify ceiling behavior; check that trend settings reach initialization; and run a minimal optimizer update. Then run the full demo and compare component metrics against the manuscript's reference results. A short-fit test alone cannot establish scientific reproducibility.

## Documentation basis

The documentation structure follows the emphasis on installation, examples, validation and input availability in the [Nature Communications computational reproducibility guidelines](https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf). This is preparation for review, not certification of compliance or manuscript acceptance.
