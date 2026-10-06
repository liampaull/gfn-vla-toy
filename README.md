# GFlowNet action decoding for a toy VLA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liampaull/gfn-vla-toy/blob/main/gfn_vla_toy.ipynb)

Quickest way to try it: the Colab badge above (the notebook needs nothing beyond torch / numpy / matplotlib,
which Colab ships with; pick a GPU runtime and "Run all", ~10 min). For a local run see below.

`gfn_vla_toy.ipynb` is a self-contained prototype of the idea "keep the VLA's vision-language backbone and
discrete action-token DAG, replace the discrete-diffusion decoder objective with a GFlowNet":

* toy task: point robot, two coloured targets, one obstacle, instruction says which target; action chunk =
  8 steps x (dx, dy), 7 bins each = 16 action tokens;
* one tiny transformer over [image patches | instruction tokens | action tokens], pre-trained on a grounding task;
* decoders compared on the same DAG of partial token assignments:
  discrete diffusion (masked CE on 85/15 left-biased demos, random or confidence order) vs. GFlowNet
  (trajectory balance on a compositional simulator reward, fixed or learned order with a learned backward policy);
* fairness baselines: discrete diffusion on 200k demos (matched data), best-of-8 under the simulator reward at
  inference, and GRPO-style policy-gradient fine-tuning of the diffusion model with the same reward and budget;
* metrics: success, clean success, collisions, wrong target, mode coverage (left/right detours), sample diversity,
  decoding-order maps, parallel-decoding degradation.

`gfn_vla_toy.py` is the same content with `# %%` cell markers (easier to diff / edit); rebuild the notebook from it with
any py-to-ipynb converter (jupytext, or the small nbformat script used here).

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
PATH=.venv/bin:$PATH jupyter-lab gfn_vla_toy.ipynb          # interactive
PATH=.venv/bin:$PATH jupyter-nbconvert --to notebook --execute --inplace gfn_vla_toy.ipynb   # headless
```

Device is picked automatically (`cuda` > `mps` > `cpu`). Whole notebook: ~15 min on Apple-silicon / GPU,
~50 min on CPU (the `FAST` flag halves the GFlowNet budget there).
