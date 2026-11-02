<h1 align="center">
  <img src="misc/art_fig.png" width="200" /></a><br>
  <b>Automated Design of Agentic Systems</b><br>
</h1>

<p align="center">
  <a href="https://github.com/ShengranHu/ADAS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2408.08435"><img src="https://img.shields.io/badge/arXiv-2408.08435-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://www.shengranhu.com/ADAS/"><img src="https://img.shields.io/badge/-Website-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://twitter.com/shengranhu/status/1825555341922480322"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

<h3 align="center" style="display:inline-block; border:2px solid red; border-radius:4px; padding:4px; margin:4px 0;">
 <strong> ICLR 2025</strong><br>
</h3>
<h3 align="center" style="display:inline-block; border:2px solid red; border-radius:4px; padding:4px; margin:4px 0;">
  🏆 <strong>Outstanding Paper (NeurIPS 2024 Open-World Agent Workshop)</strong> 
  <a href="https://x.com/shengranhu/status/1868475359060226191" target="_blank" style="margin-left:8px;">[Tweet]</a>
</h3>

In this work, we describe a newly forming research area **A**utomated **D**esign of **A**gentic **S**ystems (**ADAS**), which aims to *automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways.*


We present a simple yet effective ADAS algorithm named **Meta Agent Search** to demonstrate that **agents can invent novel and powerful agent designs**. In Meta Agent Search, a "meta" agent iteratively *programs* interesting new agents in code based on previous discoveries.


<p align="center">
<img src="misc/algo.png"/></a><br>
</p>

## Setup
```bash
conda create -n adas python=3.11
conda activate adas
pip install -r requirements.txt

# provide your OpenAI API key
export OPENAI_API_KEY="YOUR KEY HERE"
```

### Snellius: hosted Hugging Face models

Snellius provides a shared Hugging Face cache at
`/projects/2/managed_datasets/hf_cache_dir`. The setup below uses it in offline
mode, so model weights are not copied into the project or downloaded again.
See the [SURF list of available datasets and models](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/62227640/Available+datasets+and+models+on+Snellius).

Create the Conda environment once on a login node:

```bash
./scripts/create_snellius_env.sh
```

Do not run model inference on a login node. For interactive work, log in to a
GPU node first:

```bash
ssh gcn1
```

That node hands out a single A100 MIG slice (1g.5gb). It is fine for quick
checks such as `torch.cuda.is_available()`, but vLLM cannot start on a MIG
device (NVML device inspection fails), so run the pipeline itself on a full GPU
through `salloc`/`sbatch` on the `gpu_a100` partition.

Submit the included GPU job, which starts a local OpenAI-compatible vLLM server
and then runs ADAS against it:

```bash
sbatch --export=ALL,MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
  scripts/run_adas_vllm.slurm --n_generation 1 --valid_size 8 --max_workers 8
```

The default target is `_mmlu/search.py`. Select another ADAS entry point with
`ADAS_SCRIPT`, for example:

```bash
sbatch --export=ALL,MODEL=mistralai/Mistral-7B-Instruct-v0.2,ADAS_SCRIPT=_gpqa/search.py \
  scripts/run_adas_vllm.slurm --n_generation 1 --valid_size 8 --max_workers 8
```

The model must be present in the shared cache and accessible under its license.
Restricted models require access through a SURF service-desk request. Larger
models may require more GPUs and matching vLLM tensor-parallel settings. The
job exports `ADAS_MODEL`, `OPENAI_BASE_URL`, and the Hugging Face offline/cache
variables; both the meta-agent and generated agents therefore use the selected
local model. To use an already-running OpenAI-compatible endpoint instead, set
those variables manually and pass its model name through `--model`.

### GitHub Copilot models (experimental)

The same pipeline can run on models served through the
[GitHub Copilot SDK](https://github.com/github/copilot-sdk), billed in GitHub AI
Credits (1 credit = $0.01), on a local machine with no GPU. A gateway in front of
the SDK serves the OpenAI chat API that the search already uses, so the search,
sandbox and instrumentation are unchanged. The phases, gates, budget and protocol
deviations (notably: temperature cannot be set) are in
[experimental_plan_copilot.md](experimental_plan_copilot.md).

Create an environment (Python 3.11 or newer):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-copilot.txt
source .venv/bin/activate
```

Authenticate with one of these; when several are present, the SDK uses the first
in this order:

* export a fine-grained personal access token with the Copilot Requests
  permission as `COPILOT_GITHUB_TOKEN` (classic `ghp_` tokens are not supported)
* sign in with the Copilot CLI (`copilot`, then `/login`)
* sign in with the GitHub CLI (`gh auth login`)

The gateway uses the SDK's `copilot-cli` client mode by default, the only mode
that sees a stored login. Each session is deleted after use. With a token you
can pass `--client_mode empty` to isolate the runtime in `~/.cache/adas-copilot`.

Check what your account offers, run the overhead gate, then run cells:

```bash
python experiments/serving/copilot_probe.py inventory
python experiments/serving/copilot_probe.py audit --model claude-haiku-5.5
python experiments/local/run_cell.py --list
python experiments/local/run_cell.py cp0_smoke
```

Models and prices live in `experiments/config/copilot_models.yaml`, cells and
per-cell credit caps in `experiments/config/copilot_cells.yaml`, and results in
`experiments/results_copilot/`. A rerun resumes a cell. When a cell reaches its
cap, the gateway holds further requests and the runner stops the search with
exit code 3; raise the cap and rerun to continue. Keep the Mac on AC power with
the lid open: the runner holds `caffeinate -s`, which prevents idle sleep only on
AC, and closing the lid still sleeps the machine. To stop a cell, press Ctrl+C (or
send SIGINT or SIGTERM), which also stops its gateway and search; `--max_calls_per_minute N`
overrides the cell's pace on the next run. Results are in
[findings_copilot.md](findings_copilot.md).

To run several cells at once, use the parallel driver. Cells wait for the cells in
their `after:` list, and every gateway draws call starts from one account-wide pace
(`pace_rates` in `copilot_cells.yaml`; the Copilot rate limit is per account, so the
default is a single `all` budget; override it with `--run_cell_args --pace_rates all=450`),
so parallel cells share the rate limit rather than exceed it. Rerunning skips finished
cells and resumes the rest; Ctrl+C stops every running cell cleanly.

```bash
python experiments/local/run_parallel.py --group luna --dry_run
python experiments/local/run_parallel.py --group luna
```

DROP cells can also propose several designs per round from one archive snapshot
(`proposals_per_round` in the cell grid, `--proposals_per_round` for `_drop/search.py`);
the default of 1 is the ADAS loop.

> [!WARNING]
> These runs spend real AI credits. Keep the caps in `copilot_cells.yaml` and run
> the Phase 0 probes before any search.

Offline tests, no credentials or credits needed:

```bash
python experiments/tests/copilot_gateway_test.py
python experiments/tests/copilot_e2e.py
python experiments/tests/search_e2e.py gpqa
```

## Running Instructions

### Running Meta Agent Search

To run experiments for each domain, navigate to its respective folder. The code in each folder is self-contained. Launch experiments using the `search.py` script located in each domain's folder.

```bash
python {DOMAIN}/search.py
```

Replace `{DOMAIN}` with the specific domain folder name {`_arc`, `_drop`, `_mgsm`, ...} to run the experiment for.

### Customizing Meta Agent Search for New Domains

You can easily adapt the code to search for new domains. To do so, follow these steps:

1. Modify the `evaluate_forward_fn()` function and adjust any necessary formatting prompts (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/search.py#L89)) in the `search.py` file. 

2. Consider adding additional basic functions for the meta agent to utilize during the design process (similar to [this line](https://github.com/ShengranHu/ADAS/blob/main/_arc/search.py#L161)).

3. Update the domain-specific information within the prompts to match the requirements of your new domain (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/mmlu_prompt.py#L229)).

4. Run the search and evaluation on your new domain.

### Safety Consideration
> [!WARNING]  
> The code in this repository involves executing untrusted model-generated code. We strongly advise users to be aware of this safety concern. While it is highly unlikely that model-generated code will perform overtly malicious actions in our current settings and with the models we use, such code may still act destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.


## Citing
If you find this project useful, please consider citing:
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```
