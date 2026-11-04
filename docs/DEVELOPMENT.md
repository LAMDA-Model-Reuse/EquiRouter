# Development and evaluation

## Layout and interfaces

| Location | Responsibility |
| --- | --- |
| `methods/EquiRouter.py` | QMIR scoring network, CARO objective, utility training |
| `methods/base.py` | Dataset/encoder setup, embedding cache, shared evaluation |
| `methods/system_one/` | Training-only retrieval, Choice requests, backend adapters, score cache |
| `utils/cost.py` | Cost standardization, optimizer construction, shared MLP cost training |
| `utils/data.py` | Four public benchmark loaders and aligned candidate columns |
| `utils/metrics.py` | Budget sweep, QNC, peak utility, and normalized AUC |
| `configs/` | Benchmark, EquiRouter, and System One settings |
| `tests/` | CPU unit tests and synthetic integration checks |

Routers expose `train()`, `predict(embeddings) -> (scores, costs)`, and
`evaluate()`. Both returned arrays have shape `(queries, candidates)` in
`router.model_list` order. For System One, `train()` prepares retrieval and
fits the cost predictor; the decision model itself is not fine-tuned.
`predict(..., queries=frame)` accepts explicit custom input rows. Without
`queries`, embeddings must follow the test frame's row order.

## Dependencies and entry points

Use Python 3.10 or newer in a virtual environment. Install a compatible
PyTorch/torchvision pair for the target hardware before other dependencies.

```bash
python -m pip install -r requirements.txt
python main.py --dataset routerbench --device auto
python -m scripts.reproduce_equirouter --dataset routerbench --device auto
```

MMR-Bench retains missing targets from the upstream data. The current training
and evaluation entry points require finite targets and stop when missing values
are encountered.

Configuration files resolve relative to the source tree. Data, caches, logs,
and output paths resolve relative to the working directory, or to the paths
provided by the caller. Hyperparameters are loaded from
`configs/routers/EquiRouter.json`; dataset overrides are case-insensitive.

MiniLM resolves `sentence-transformers/all-MiniLM-L6-v2` through the standard
Hugging Face cache. Set `embeddings.text_model_dir` to a local checkpoint for
offline use. MMR-Bench uses OpenAI CLIP `ViT-B/16`. Encoder imports are lazy;
text-only use does not initialize CLIP. Runtime encoders remain frozen.

## Benchmark sources

| Benchmark | Public source | Input |
| --- | --- | --- |
| RouterBench | [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench) | Text |
| MMR-Bench | [gh0stHunter/MMR-Bench](https://huggingface.co/datasets/gh0stHunter/MMR-Bench) | Text and image |
| MixInstruct | [llm-blender/mix-instruct](https://huggingface.co/datasets/llm-blender/mix-instruct) | Text |
| RouterEval | [linggm/Routereval](https://huggingface.co/datasets/linggm/Routereval) | Text |

Source revisions and SHA-256 values are recorded in [`utils/sources.py`](../utils/sources.py).
Downloads use those revisions; local copies of the binary assets are also verified.
RouterBench is converted to a local Parquet cache after the initial verified
Pickle import. RouterEval verifies both the ZIP and each of the six task Pickles
before deserialization. These hashes identify trusted upstream files; they do
not make arbitrary Pickle files safe to load.

| Benchmark | Utility | Cost | Candidate handling |
| --- | --- | --- | --- |
| RouterBench | Published per-query performance values | Published per-query costs | The 11 named candidates in the loader, aligned by sample ID; missing values remain missing. |
| MMR-Bench | Published correctness values converted to 0/1 | Published per-query costs | The 10 named candidates in the loader; each reads only its own columns. `gpt-5-minimal` is not substituted for `gpt-5-nano`. Missing utilities, costs, and responses remain missing; no rows are dropped. |
| MixInstruct | Continuous BERTScore | Model parameter counts divided by the largest candidate's count | The 12 named candidates; upstream splits are combined before the configured split. |
| RouterEval | Published task scores | Model sizes parsed from names, followed by `log1p` and per-row normalization | The sorted intersection of size-parsable candidates across the six Group A tasks. |

RouterEval's normalization maps `x = log1p(size)` to
`0.1 + 0.8 * (x - (min(x) - 0.1*r)) / (1.2*r)`, where
`r = max(x) - min(x)` within each query. Equal-size rows map to 0.5.
MixInstruct and RouterEval costs are size-based proxies.

The default split is 40% training and 60% testing, seed 42. Both ratios must be
explicit, positive, and sum to one. In-domain splitting shuffles once and uses
`floor(N * train_ratio)` training rows; all remaining rows form the test set.
Task-grouped out-of-domain splitting keeps tasks intact, so the realized row
ratio can differ from the requested target. Train/test utilities and costs use
the same candidate order. Training and evaluation require finite targets and
report an error when raw missing values are present. Corrupt or missing images
also raise an error.

## System One baselines

The implementation follows Appendix A.2 of the final manuscript:

1. Retrieve 32 distinct training examples by cosine similarity in the shared
   query embedding space. Exclude normalized target-text duplicates and shared
   image-content duplicates, as well as duplicates within the retrieved history.
   Fewer than 32 eligible examples cause an explicit error. `neighbors` can be
   changed for small development datasets; this changes the paper protocol.
2. Construct a Choice request containing the target input, candidate identities,
   and the candidates' observed utilities on retrieved training examples. For
   multimodal routing, attach the target image and each history image. No costs,
   target labels, cached candidate responses, or local file names enter the request.
3. Validate the returned probability vector and restore canonical candidate
   order. Requests are hashed with the backend model/version; validated scores
   are reused across calls and budget points. Cache files contain only option
   probabilities, not requests or credentials.
4. Fit the shared cost predictor on training data and apply the same budget
   evaluation as EquiRouter. Test utilities and actual costs remain offline
   evaluation targets.

The modules have separate boundaries: `retrieval.py`, `schema.py`, `backends.py`,
`cache.py`, and `router.py`. A custom backend supplies `model`, `cache_identity`,
and `predict(request)`, returning `answers.selected_model` with a `choice` and
complete `probabilities` map.

### Jev 1.14

```bash
python main.py --dataset routerbench --method jev --device cpu
```

Set `TYPESAFE_API_KEY` in the process environment. `TYPESAFE_ENDPOINT` optionally
overrides the public System One endpoint. Endpoints must use HTTPS. Authenticated
redirects to another origin and redirects to HTTP are rejected. Keys are never written to configuration files,
request bodies, caches, or logs. The default model identifier is `jev-1.14.0`.
The adapter uses the [TypeSafe System One interface](https://docs.typesafe.ai/api/system-one)
and its [image asset schema](https://docs.typesafe.ai/images); model availability
and image support depend on the account and service. Transient HTTP failures
receive bounded retries; malformed decisions fail rather than falling back to
another router. Running this backend sends benchmark query/image content to
the configured service and may incur API charges.

### Clef-Flash

```bash
python -m pip install -r requirements-clef.txt
python main.py --dataset routerbench --method clef-flash --device cuda:0
```

The native backend lazily loads the publisher's schema head and processor from
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), pinned to
revision `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`. Set `CLEF_MODEL_DIR` to a
complete local snapshot, or allow the first inference call to download that
revision. The optional requirements provide Transformers' Qwen3.5 implementation
and Accelerate. A suitable GPU with sufficient memory is needed for practical
9B-model inference; this code does not supply model weights or serving-specific
multi-GPU memory optimizations.

Complete evidence is encoded before inference. Inputs exceeding the backbone's
context limit (or configured `max_length`) are rejected, never silently
truncated. Local checkpoint overrides have separate cache namespaces; clear the
score cache if changing weights in the same local directory. Backend adapters
load only when used, so ordinary EquiRouter tests require no decision checkpoint
or API key.

## Metrics and reproduction scope

`evaluate_budget_curve` uses 100 thresholds from the smallest budget feasible
for every query to the maximum predicted cost. It selects the highest-scoring
feasible candidate, evaluates its observed utility/cost, and reports peak
utility, QNC, target-level savings, and nAUC on shared cost bounds. QNC is
undefined if no evaluated point reaches the best-single-model quality target,
or if that target model has zero cost. Both evaluation entry points clip
predicted costs to training-observed candidate bounds.

The network contains one QMIR utility-scoring network and a separate MLP cost
predictor. README comparison values and figures are attributed to the paper.
The reproduction entry point can save metrics and provenance to its output
directory. Run records under the default `outputs/` directory are excluded from
the source release. Review custom output directories before packaging.

## Validation

```bash
python -m pip install 'pytest>=8,<10' 'ruff>=0.11,<1'
python -m ruff check .
python -m ruff format --check .
python -m compileall -q main.py train.py methods scripts utils tests
python -m pip check
python -m pytest -q
```

Tests use synthetic inputs for local training, multimodal request construction,
retrieval, caching, budget evaluation, and backend contracts. They do not invoke
paid services or download decision weights. Full benchmark runs and native 9B inference are separate from unit tests.

## Source release

Keep local data, virtual environments, caches, and experiment outputs outside the
source archive. The existing `.gitignore` excludes the default runtime paths.
Create an archive from Git-visible source files with:

```bash
python -m scripts.package_release --output outputs/EquiRouter-source.tar.gz
```

The packager also excludes runtime directories and generated binary files,
including files that were accidentally tracked.
