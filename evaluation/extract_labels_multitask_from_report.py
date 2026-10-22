import os, ast, argparse, sys, json, time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import re
from openai import OpenAI
from .prompts import PROMPTS, DX_ALLOWED, DX_COMPONENTS_ALLOWED

MODEL = "MODEL_NAME"  # default model name for logging purposes
DEFAULT_OUT_DIR = "classification/results"


def report_id_from_path(fp: str):
    base = os.path.splitext(os.path.basename(fp))[0]
    r_split_1 = f"Report_{base}-1_1"
    r_split_2 = f"Report_{base}-1"
    return r_split_1, r_split_2, base


def safe_json_loads(s: str):
    clean = re.sub(r"^```json\s*|\s*```$", "", s.strip())
    return json.loads(clean)


def compute_scores(y_true, y_pred, labels=None):
    """
    labels: classes over which macro metrics are averaged.
    Pass the GT classes (sorted(set(y_true))): predictions of classes absent from the GT
    (e.g. -1 when the generated report omits the feature) still count as errors for the
    true class, but are NOT added as an extra class with score 0 to the macro average.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # Normalizza casi tipo "Diagnosis,invasive ductal carcinoma"
    if s.lower().startswith("diagnosis,"):
        s = s.split(",", 1)[1].strip()
    return s


def load_gt_multitask(gt_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    """
    Ritorna:
      - gt_map: report_id -> task_name -> label_int
      - text2label: task_name -> text -> label_int (costruito dal GT)
      - label2text: task_name -> label_int -> text
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map: Dict[str, Dict[str, int]] = {}
    text2label: Dict[str, Dict[str, int]] = defaultdict(dict)
    label2text: Dict[str, Dict[int, str]] = defaultdict(dict)

    for item in data:
        rid = item["id"]
        tasks = item.get("tasks", {})
        gt_map[rid] = {}

        for tname, tinfo in tasks.items():
            lab = int(tinfo["label"])
            txt = normalize_text(tinfo.get("text", ""))

            gt_map[rid][tname] = lab

            if tname == 'diagnosis' and lab == 2:
                txt = 'other'

            if txt:
                # costruisci mapping text<->label dal GT
                if txt in text2label[tname] and text2label[tname][txt] != lab:
                    pass
                else:
                    if "not specified" in txt.lower():
                        txt = txt.replace("not specified", "unspecified")
                    if tname == 'grade' and lab == -1:  # Unknown -> Grade (unspecified)
                        txt = 'Grade (unspecified)'
                    text2label[tname][txt] = lab

                if lab in label2text[tname] and label2text[tname][lab] != txt:
                    pass
                else:
                    if "not specified" in txt.lower():
                        txt = txt.replace("not specified", "unspecified")
                    if tname == 'grade' and lab == -1:  # Unknown -> Grade (unspecified)
                        txt = 'Grade (unspecified)'
                    label2text[tname][lab] = txt

    return gt_map, text2label, label2text


def build_tasks_config():
    """
    output_key = chiave JSON che il GPT deve restituire
    sys/user = prompt RESULTS se esiste
    """
    return {
        "diagnosis": {
            "system_prompt_key": "BRCA_DX_SYSTEM_PROMPT",
            "user_prompt_key": "BRCA_DX_USER_PROMPT",
            "output_key": "diagnosis",
            "extra_format_kwargs": {
                "allowed_labels": "\n".join(f"- {x}" for x in DX_ALLOWED),
                "allowed_components": "\n".join(f"- {x}" for x in DX_COMPONENTS_ALLOWED),
            },
        },
        "grade": {
            "system_prompt_key": "BRCA_GRADE_SYSTEM_PROMPT",
            "user_prompt_key": "GRADE_USER_PROMPT",
            "output_key": "grade",
        },
        "nuclear_pleomorphism": {
            "system_prompt_key": "BRCA_NUCLEAR_PLEOMORPHISM_SYSTEM_PROMPT",
            "user_prompt_key": "NUCLEAR_PLEOMORPHISM_USER_PROMPT",
            "output_key": "nuclear_pleomorphism",
        },
        "mitotic_activity": {
            "system_prompt_key": "BRCA_MITOTIC_ACTIVITY_SYSTEM_PROMPT",
            "user_prompt_key": "MITOTIC_ACTIVITY_USER_PROMPT",
            "output_key": "mitotic_activity",
        },
        "tubular_formation": {
            "system_prompt_key": "BRCA_GLANDULAR_TUBULAR_FORMATION_SYSTEM_PROMPT",
            "user_prompt_key": "GLANDULAR_TUBULAR_FORMATION_USER_PROMPT",
            "output_key": "glandular_tubular_formation",
        },
        "LVI": {
            "system_prompt_key": "VASCULAR_LYMPHOVASCULAR_INVASION_SYSTEM_PROMPT",
            "user_prompt_key": "VASCULAR_LYMPHOVASCULAR_INVASION_USER_PROMPT",
            "output_key": "vascular_lymphovascular_invasion",
        },
    }


def extract_with_gpt(client: OpenAI, model_type: str, cfg: Dict[str, Any], report_text: str, allowed_classes: Optional[str]):
    sys_prompt = PROMPTS[cfg["system_prompt_key"]]
    user_template = PROMPTS[cfg["user_prompt_key"]]

    fmt_kwargs = dict(cfg.get("extra_format_kwargs", {}))
    fmt_kwargs["report_text"] = str(report_text)

    if "{allowed_classes}" in user_template and allowed_classes is not None:
        fmt_kwargs["allowed_classes"] = allowed_classes

    raw_user = user_template.format(**fmt_kwargs)

    resp = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": raw_user},
        ],
    )
    raw = resp.choices[0].message.content
    obj = safe_json_loads(raw)
    obj["_raw"] = raw
    return obj


def main():
    ap = argparse.ArgumentParser("Multitask GPT extraction + metrics vs GT (single JSON)")
    ap.add_argument("--root", required=True, help="Folder containing *.txt predictions")
    ap.add_argument("--gt", required=True, help="GT multitask json")
    ap.add_argument("--out_txt", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--use_target_text", action="store_true")
    ap.add_argument("--model_type", default="gpt-5")
    ap.add_argument("--model", default=MODEL, help="Model name for logging purposes")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true", help="Limit to 20 files for debugging")
    ap.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'")
    ap.add_argument("--save_name", type=str, default='', help="Split name for logging purposes")
    ap.add_argument("--seed", type=int, default=9233, help="Seed for logging purposes (e.g. 9233)")
    args = ap.parse_args()

    if args.out_txt is None:
        args.out_txt = os.path.join(DEFAULT_OUT_DIR, args.model, f"metrics_{args.save_name}_multitask_gpt.txt")
    if args.out_csv is None:
        args.out_csv = os.path.join(DEFAULT_OUT_DIR, args.model, f"preds_{args.save_name}_multitask_gpt.csv")

    cache_dir_output = os.path.join(DEFAULT_OUT_DIR, args.model, f"gpt_cache_{args.save_name}_multitask")
    os.makedirs(cache_dir_output, exist_ok=True)

    gt_map, text2label, label2text = load_gt_multitask(args.gt)
    tasks_cfg = build_tasks_config()
    valid_diagnoses = [
        "invasive ductal carcinoma",
        "invasive lobular carcinoma"
    ]

    if args.tasks != "all":
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        tasks_cfg = {k: v for k, v in tasks_cfg.items() if k in wanted}

    for t in tasks_cfg:
        os.makedirs(os.path.join(cache_dir_output, t), exist_ok=True)

    client = OpenAI(api_key=args.api_key)

    files = []
    with os.scandir(args.root) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".txt"):
                files.append(entry.path)
                if args.debug and len(files) == 20:
                    break

    # redirect log
    orig_stdout = sys.stdout
    with open(args.out_txt, "w", encoding="utf-8") as fout:
        sys.stdout = fout

        print("=== Multitask GPT extraction + metrics ===")
        print("Timestamp:", datetime.now())
        print("ROOT:", args.root)
        print("GT:", args.gt)
        print("use_target_text:", args.use_target_text)
        print("OpenAI model:", args.model_type)
        print("Cache:", cache_dir_output)
        print("Tasks:", list(tasks_cfg.keys()))
        print("Num files:", len(files))
        print()

        rows = []
        n_parse_err = n_empty = n_missing_gt = 0
        n_cached = n_called = n_gpt_err = n_map_err = 0

        per_task_true = defaultdict(list)
        per_task_pred = defaultdict(list)

        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

            try:
                obj = ast.literal_eval(content)
            except Exception:
                n_parse_err += 1
                continue

            rid1, rid2, base = report_id_from_path(fp)
            rid = rid1 if rid1 in gt_map else rid2 if rid2 in gt_map else None
            if rid is None:
                n_missing_gt += 1
                continue

            text = obj.get("target" if args.use_target_text else "predict", "")
            if not text:
                n_empty += 1
                continue

            for task_name, cfg in tasks_cfg.items():
                # Allowed classes list per task
                allowed_list = sorted(text2label.get(task_name, {}).keys())
                allowed_classes = "\n".join(f"- {x}" for x in allowed_list) if allowed_list else None

                cache_path = os.path.join(cache_dir_output, task_name, f"{base}.json")

                try:
                    true_label = gt_map[rid].get(task_name, None)

                    # 1) If GT is missing -> skip (not assessable)
                    if true_label is None:
                        rows.append({
                            "report_id": rid, "base": base, "task": task_name,
                            "gt_label": None, "gt_text": None,
                            "pred_label": None, "pred_text": None,
                            "confidence": None,
                            "evidence": None,
                            "error": "GT_LABEL_MISSING",
                        })
                        continue

                    skip_eval = (int(true_label) == -1)

                    # ----- GPT extraction with cache -----
                    if os.path.exists(cache_path):
                        with open(cache_path, "r", encoding="utf-8") as cf:
                            gpt_obj = json.load(cf)
                        n_cached += 1
                    else:
                        gpt_obj = extract_with_gpt(client, args.model_type, cfg, text, allowed_classes)
                        with open(cache_path, "w", encoding="utf-8") as cf:
                            json.dump(gpt_obj, cf, ensure_ascii=False, indent=2)
                        n_called += 1
                        if args.sleep_s > 0:
                            time.sleep(args.sleep_s)

                    out_key = cfg["output_key"]
                    pred_text = normalize_text(gpt_obj.get(out_key, ""))

                    # mapping testo -> label
                    t2l = text2label.get(task_name, {})

                    if task_name == 'diagnosis':
                        # unspecified invasive carcinoma -> IDC
                        if pred_text == "invasive carcinoma (unspecified)":
                            pred_text = "invasive ductal carcinoma"
                        elif not any(diagnosis in pred_text for diagnosis in valid_diagnoses):
                            pred_text = "other"

                    if pred_text not in t2l:
                        n_map_err += 1
                        pred_label = None
                    else:
                        pred_label = t2l[pred_text]

                    # 2) If pred_label = None -> mapping fallito: non valutare
                    if pred_label is None:
                        rows.append({
                            "report_id": rid, "base": base, "task": task_name,
                            "gt_label": int(true_label),
                            "gt_text": label2text.get(task_name, {}).get(int(true_label), None),
                            "pred_label": None,
                            "pred_text": pred_text,
                            "confidence": gpt_obj.get("confidence", None),
                            "evidence": gpt_obj.get("evidence", None),
                            "error": f"PRED_TEXT_NOT_IN_ALLOWED: {pred_text}",
                        })
                        continue

                    # 3) If GT == -1: logged but not used in the metric computation
                    if skip_eval:
                        rows.append({
                            "report_id": rid, "base": base, "task": task_name,
                            "gt_label": true_label,
                            "gt_text": label2text.get(task_name, {}).get(true_label, None),
                            "pred_label": pred_label,
                            "pred_text": pred_text,
                            "confidence": gpt_obj.get("confidence", None),
                            "evidence": gpt_obj.get("evidence", None),
                            "error": "SKIP_GT_-1",
                        })
                        continue

                    # ----- Evaluation -----
                    # NOTE: a prediction of -1 (feature omitted in the generated report) is kept and
                    # counted as an error for the true class.
                    per_task_true[task_name].append(true_label)
                    per_task_pred[task_name].append(pred_label)

                    rows.append({
                        "report_id": rid, "base": base, "task": task_name,
                        "gt_label": int(true_label),
                        "gt_text": label2text.get(task_name, {}).get(int(true_label), None),
                        "pred_label": int(pred_label),
                        "pred_text": pred_text,
                        "confidence": gpt_obj.get("confidence", None),
                        "evidence": gpt_obj.get("evidence", None),
                        "error": None if int(pred_label) == int(true_label) else "WRONG",
                    })

                except Exception as e:
                    n_gpt_err += 1
                    rows.append({
                        "report_id": rid, "base": base, "task": task_name,
                        "gt_label": gt_map[rid].get(task_name, None),
                        "gt_text": None,
                        "pred_label": None,
                        "pred_text": None,
                        "confidence": None,
                        "evidence": None,
                        "error": f"GPT_ERR: {repr(e)}",
                    })

        # save CSV
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False)

        print("\n--- Global summary ---")
        print("Parse errors:", n_parse_err)
        print("Missing in GT:", n_missing_gt)
        print("Empty texts:", n_empty)
        print("GPT errors:", n_gpt_err)
        print("Mapping errors (pred text not in allowed):", n_map_err)
        print("Cache hits:", n_cached)
        print("GPT calls:", n_called)
        print("CSV rows:", len(df))
        print("Saved CSV:", args.out_csv)
        print()

        # metrics per task
        for task_name in tasks_cfg.keys():
            y_true = per_task_true[task_name] if task_name in per_task_true else []
            y_pred = per_task_pred[task_name] if task_name in per_task_pred else []
            print("=" * 80)
            print(f"Task: {task_name} | evaluated={len(y_true)}")

            if len(y_true) == 0:
                print("[WARN] No samples evaluated for this task (check mapping / GT presence).")
                continue

            # all labels (GT + predictions, including -1) -> only for display
            labels_sorted = sorted({*y_true, *y_pred})
            target_names = [label2text.get(task_name, {}).get(l, str(l)) for l in labels_sorted]

            # GT classes only -> used for macro metrics
            labels_eval = sorted(set(y_true))

            n_omitted = sum(1 for p in y_pred if p == -1)
            print("Labels (display):", labels_sorted)
            print("Labels (metrics):", labels_eval)
            print("Predictions = -1 (feature omitted, counted as errors):", n_omitted)
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred, labels=labels_sorted))

            print("\nClassification report:")
            print(classification_report(
                y_true, y_pred,
                labels=labels_sorted,
                target_names=target_names,
                digits=4,
                zero_division=0
            ))

            scores = compute_scores(y_true, y_pred, labels=labels_eval)
            print("Metrics:")
            for k, v in scores.items():
                print(f"  {k:15s}: {v:.4f}")

        sys.stdout = orig_stdout

    print(f"✅ Salvati: {args.out_txt} e {args.out_csv}")


if __name__ == "__main__":
    main()