# Feature Extraction

Patch- and slide-level feature extraction for PathoClass-BRCA. Each model family consumes a
different encoder, so this folder holds three independent extraction scripts, plus one that builds
the text knowledge bank for BiGen. The three extraction scripts all write one `.pt` per slide to
`<OUT_ROOT>/BRCA/<slide_id>.pt`.

| Script | Encoder | Output | Passed to training as |
|---|---|---|---|
| `extract_UNI_embeddings_WSI.py` | [UNI](https://github.com/mahmoodlab/UNI) | `(N, 1024)` patch embeddings | `--image_dir` |
| `extract_PLIP_embeddings_WSI.py` | [PLIP](https://github.com/PathologyFoundation/plip) | `(N, 512)` L2-normalised patch embeddings | `--image_dir_plip` |
| `extract_gigapath_embeddings.py` | [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) | one slide embedding, zero-padded to 1024 | — (WSI-LLaVA only) |
| `extract_sentences_wsi_bench.py` | PLIP (text) | `(T, 512)` L2-normalised text bank | `--bank_path` |

## What you need

| Training… | UNI | PLIP patches | Text bank | GigaPath |
|---|:---:|:---:|:---:|:---:|
| MiGen / MiGen† | ✅ | — | — | — |
| HistGen / HistGen† | ✅ | — | — | — |
| BiGen / BiGen† | ✅ | ✅ | ✅ | — |
| WSI-LLaVA (evaluation only) | — | — | — | ✅ |

## Conventions

These scripts consume the patches produced by the WSI pre-processing step in the
[main README](../README.md#2-whole-slide-image-processing).

- **Layout:** `<TCGA_ROOT>/BRCA/images/<slide_id>/*.jpg`
- **Filenames:** patch coordinates as `_x_<X>_y_<Y>.jpg`. UNI and PLIP sort by them in raster-scan
  order; GigaPath passes them to the slide encoder as spatial positions. A non-matching filename
  raises an error.
- **Slide list CSV:** decides which slides get encoded. Three columns are read: `cancer` (the
  cohort, `BRCA`), `slide_base` (must equal the `<slide_id>` folder name holding that slide's
  patches) and `status` (only rows whose value is exactly `FOUND` are processed).

---

## UNI — patch embeddings for all backbones

```bash
python extract_embeddings/extract_UNI_embeddings_WSI.py \
    --images_root <TCGA_ROOT> \
    --csv <SLIDE_LIST_CSV> \
    --out_root <PT_DIR> \
    --cancers BRCA \
    --uni_path <UNI_ASSETS_DIR> \
    --skip_existing
```

`--uni_path` is the local directory holding the UNI checkpoint and config.

---

## PLIP — retrieval features for BiGen

BiGen ranks patches and retrieves related report embeddings in the PLIP joint image–text space, so it needs this set **alongside** the UNI one. Embeddings are L2-normalised before saving, as the cosine-similarity retrieval requires.

```bash
python -m extract_embeddings.extract_PLIP_embeddings_WSI \
    --images_root <TCGA_ROOT> \
    --csv <SLIDE_LIST_CSV> \
    --out_root <PLIP_PT_DIR> \
    --cancers BRCA \
    --plip_path <PLIP_CHECKPOINT> \
    --skip_existing
```

> Run it with `python -m`, not as a file path: the script uses a package-relative import for the
> bundled PLIP wrapper.

---

## GigaPath — slide embeddings for WSI-LLaVA

```bash
python extract_embeddings/extract_gigapath_embeddings.py \
    --images_root <TCGA_ROOT> \
    --csv <SLIDE_LIST_CSV> \
    --out_root <GIGAPATH_PT_DIR> \
    --tile_model_path <TILE_ENCODER_CKPT>                     `# Checkpoint for the tile encoder.`  \     
    --slide_model_path <SLIDE_ENCODER_CKPT>                   `# Checkpoint for the LongNet slide encoder.`   \
    --cancers BRCA \
    --fp16
```

---

## Text knowledge bank for BiGen

BiGen retrieves related text embeddings as auxiliary context during generation. This script splits
the training reports into individual sentences, encodes each one with the PLIP text encoder, and
saves the L2-normalised result as a single `(T, 512)` tensor, where `T` is the total number of
sentences. It is built once and reused across runs.

```bash
python -m extract_embeddings.extract_sentences_wsi_bench \
    --input_file dataset/report_generation/BRCA/WSI-Bench-train-report.json \
    --output_sentences dataset/report_generation/BRCA/short_wsi_bench.json \
    --output_embeddings memory_short.pt \
    --plip_checkpoint <PLIP_CHECKPOINT>
```

> Build the bank from the **training** split only, so no test-set text reaches the retrieval index.
