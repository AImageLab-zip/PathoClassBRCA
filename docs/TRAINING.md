# Training

This page gives a complete command for every model and paradigm.

All experiments are launched through `main.py`. The same entry point handles both
paradigms: adding `--classification` switches from free-text report generation to
multi-task classification, selecting the `*_classification.py` variant of the chosen
backbone.

> **Prerequisite:** extract the UNI / PLIP / Gigapath embeddings before running any of
> the models below — see [feature extraction](FEATURE_EXTRACTION.md).


- [Training](#training)
  - [MiGen](#migen)
    - [Classification (MiGen†)](#classification-migen)
    - [Report generation](#report-generation)
  - [HistGen](#histgen)
    - [Classification (HistGen†)](#classification-histgen)
    - [Report generation](#report-generation-1)
  - [BiGen](#bigen)
    - [Classification (BiGen†)](#classification-bigen)
    - [Report generation](#report-generation-2)
  - [WSI-LLaVA](#wsi-llava)
  - [Testing a checkpoint](#testing-a-checkpoint)
  - [Reproducing the paper](#reproducing-the-paper)

---

## MiGen

### Classification (MiGen†)

```bash
python main.py \
    --mode Train \
    --classification \
    --model_name migen \
    --image_dir <PT_DIR> \
    --train_json dataset/classification/BRCA/splits/train_multitask.json \
    --val_json   dataset/classification/BRCA/splits/val_multitask.json \
    --test_json  dataset/classification/BRCA/splits/test_multitask.json \
    --tasks diagnosis grade nuclear_pleomorphism mitotic_activity tubular_formation LVI \
    --monitor_metric f1_macro \
    --num_prompts_per_task 1 \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

> `--num_prompts_per_task` is $l_{tp}$ in the paper; **HistGen and BiGen currently support `1` only**, while **MiGen also accepts larger values** (ablated in Table 2).


### Report generation

```bash
python main.py \
    --mode Train \
    --model_name migen \
    --image_dir <PT_DIR> \
    --train_json dataset/report_generation/BRCA/WSI-Bench-train-report.json \
    --val_json   dataset/report_generation/BRCA/WSI-Bench-val-report.json \
    --test_json  dataset/report_generation/BRCA/WSI-Bench-test-report.json \
    --use_json_dataset \
    --masking_fixed \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

---

## HistGen

HistGen adds a hierarchical local–global visual encoder and a cross-modal memory module, configured by `--region_size`, `--prototype_num`, `--topk`, `--cmm_size` and `--cmm_dim`.

### Classification (HistGen†)

```bash
python main.py \
    --mode Train \
    --classification \
    --model_name histgen \
    --image_dir <PT_DIR> \
    --train_json dataset/classification/BRCA/splits/train_multitask.json \
    --val_json   dataset/classification/BRCA/splits/val_multitask.json \
    --test_json  dataset/classification/BRCA/splits/test_multitask.json \
    --tasks diagnosis grade nuclear_pleomorphism mitotic_activity tubular_formation LVI \
    --monitor_metric f1_macro \
    --num_prompts_per_task 1 \
    --cmm_dim 512 \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

### Report generation

```bash
python main.py \
    --mode Train \
    --model_name histgen \
    --image_dir <PT_DIR> \
    --train_json dataset/report_generation/BRCA/WSI-Bench-train-report.json \
    --val_json   dataset/report_generation/BRCA/WSI-Bench-val-report.json \
    --test_json  dataset/report_generation/BRCA/WSI-Bench-test-report.json \
    --use_json_dataset \
    --masking_fixed \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size 96 \
    --prototype_num 512 \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

---

## BiGen

BiGen retrieves textual context from an external knowledge bank, so it needs two extra inputs on **both** paradigms: the PLIP text bank (`--bank_path`) and the PLIP patch embeddings used to query the bank (`--image_dir_plip`). Build both as described in the [feature extraction README](./FEATURE_EXTRACTION.md).

Retrieval behaviour is controlled by `--v` (fraction of top-ranked knowledge features kept per region), `--m` (region size) and `--k` (top-k features by attention score), left at their defaults `0.4`, `10` and `3`.

### Classification (BiGen†)

```bash
python main.py \
    --mode Train \
    --classification \
    --model_name bigen \
    --image_dir <PT_DIR> \
    --image_dir_plip <PLIP_PT_DIR> \
    --bank_path <BANK_PT> \
    --train_json dataset/classification/BRCA/splits/train_multitask.json \
    --val_json   dataset/classification/BRCA/splits/val_multitask.json \
    --test_json  dataset/classification/BRCA/splits/test_multitask.json \
    --tasks diagnosis grade nuclear_pleomorphism mitotic_activity tubular_formation LVI \
    --monitor_metric f1_macro \
    --num_prompts_per_task 1 \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

### Report generation

```bash
python main.py \
    --mode Train \
    --model_name bigen \
    --image_dir <PT_DIR> \
    --image_dir_plip <PLIP_PT_DIR> \
    --bank_path <BANK_PT> \
    --train_json dataset/report_generation/BRCA/WSI-Bench-train-report.json \
    --val_json   dataset/report_generation/BRCA/WSI-Bench-val-report.json \
    --test_json  dataset/report_generation/BRCA/WSI-Bench-test-report.json \
    --use_json_dataset \
    --masking_fixed \
    --save_dir <OUTPUT_DIR> \
    --seed 9233 \
    --d_model 512 \
    --batch_size 1 \
    --epochs 60 \
    --num_workers <NUM_WORKERS> \
    --n_gpu <NUM_GPUS>
```

---

## WSI-LLaVA

WSI-LLaVA is **not** retrained: it is already trained on WSI-Bench, so we run its released checkpoint on the BRCA test split and score the resulting reports with the same feature-level protocol. There is no classification-oriented variant, since its only learnable visual module is a linear projector (Section 3.3 of the paper).

Inference is launched from the vendored LLaVA codebase:

```bash
python models/WSI_LLAVA/llava/eval/model_vqa.py \
    --model-path <WSI_LLAVA_CKPT> \
    --image-folder <GIGAPATH_PT_DIR> \
    --question-file dataset/report_generation/BRCA/WSI-Bench-test-report.json \
    --answers-file <ANSWERS_JSONL> \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num_beams 1 \
    --seed 9233
```

> Use `--temperature 0` for greedy decoding. With the default `0.2` the generated reports vary between runs despite `--seed`.

---

## Testing a checkpoint

Training already evaluates the final and best checkpoints on the test split. To re-run evaluation separately, point `--checkpoint_dir` at the timestamped run directory and keep the model arguments identical to training:

```bash
python main.py \
    --mode Test \
    --classification \
    --model_name migen \
    --checkpoint_dir <RUN_DIR> \
    --image_dir <PT_DIR> \
    --test_json dataset/classification/BRCA/splits/test_multitask.json \
    --tasks diagnosis grade nuclear_pleomorphism mitotic_activity tubular_formation LVI \
    --d_model 512 \
    --num_prompts_per_task 1
```

`model_best.pth` is loaded by default; add `--resume_last` to evaluate `current_checkpoint.pth` instead. Add `--save_attn_scores` to dump the attention from each learnable task prompt to the WSI patches, used for the visualisations in Figure 6.

---

## Reproducing the paper

Table 1 averages **10 seeds, 9233 to 9242**, for each of the six commands above, with statistical significance assessed by one-sided paired t-tests against the generative counterpart. WSI-LLaVA is evaluated once.
