# Third-party sources and licenses

EquiRouter code is released under the MIT License, copyright 2026 EquiRouter
authors. The shared routing and data infrastructure derives from
[ORBIT](https://github.com/LAMDA-Model-Reuse/ORBIT/tree/494810de2605f69737e72b55baf6e60c95c6dec0),
copyright 2026 LAMDA-ORBIT, also under MIT. The complete upstream notice is
preserved in [third_party/ORBIT-LICENSE](third_party/ORBIT-LICENSE).

## Benchmark data

Data are downloaded from upstream and are not included in this repository or
the source release archive. Fixed revisions and binary-file SHA-256 values are
listed in [utils/sources.py](utils/sources.py). Licenses below describe upstream
declarations; component tasks and model-generated responses can have additional
terms. The EquiRouter MIT license does not relicense those assets.

| Dataset | Upstream source | Declared license |
| --- | --- | --- |
| RouterBench | [withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench/tree/784021482c3f320c6619ed4b3bb3b41a21424fcb) | No dataset license declared in the pinned dataset card; redistribution permission is not asserted. The separate RouterBench software license is not a dataset license. |
| MMR-Bench | [gh0stHunter/MMR-Bench](https://huggingface.co/datasets/gh0stHunter/MMR-Bench/tree/b2619d8389935171022f0a7e8bee45f80ca249dd) | Apache-2.0 in the dataset card; consult the original image/task sources for component terms. |
| MixInstruct | [llm-blender/mix-instruct](https://huggingface.co/datasets/llm-blender/mix-instruct/tree/c7f77a4ef0515a99d1752c6387c22366d6922da6) | MIT in the dataset card; underlying instruction sources retain their terms. |
| RouterEval | [linggm/Routereval](https://huggingface.co/datasets/linggm/Routereval/tree/0ee4de2661854b588474c211bebe97dc1e434237) | MIT in the dataset card; component benchmarks retain their terms. |

## Pretrained models and external services

| Asset | Source and version | License or terms |
| --- | --- | --- |
| MiniLM sentence encoder | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) | Apache-2.0 |
| CLIP code and ViT-B/16 encoder | [OpenAI CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6) | MIT; code is pinned and the publisher's weight URL contains the checkpoint SHA-256. |
| Clef-Flash model and native inference code | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash/tree/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f) | Apache-2.0; downloaded on demand, including the publisher's Python inference module. |
| Jev 1.14 | [TypeSafe](https://docs.typesafe.ai/api/system-one), model `jev-1.14.0` | Proprietary external service; access, processing, and billing are governed by the user's service agreement. |

Model checkpoints and API credentials are not redistributed. Jev requests send
the selected benchmark inputs and retrieved training evidence to the service.
Paper figure sources and authorship are recorded in [assets/FIGURES.md](assets/FIGURES.md).

## Python dependencies

These packages are installed separately. Their licenses and notices remain with
their distributions; consult the linked upstream license for the installed version.

| Package | Upstream license |
| --- | --- |
| PyTorch | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| datasets | [Apache-2.0](https://github.com/huggingface/datasets/blob/main/LICENSE) |
| huggingface-hub | [Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| sentence-transformers | [Apache-2.0](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE) |
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | [BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| PyArrow | [Apache-2.0](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| Pillow | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| tqdm | [MPL-2.0 and MIT](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| Transformers (optional) | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| Accelerate (optional) | [Apache-2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE) |
| safetensors (optional) | [Apache-2.0](https://github.com/huggingface/safetensors/blob/main/LICENSE) |
