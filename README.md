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
