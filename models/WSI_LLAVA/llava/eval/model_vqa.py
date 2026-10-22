import argparse
import torch
import os
import json
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
import math
import random
import numpy as np


# Candidate keys holding the ground-truth report, tried in order when --answer-key is "auto".
ANSWER_KEY_CANDIDATES = ("T-answer", "text", "answer")


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def init_seeds(seed=999, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # opzionale (più strict, può rallentare o dare errori se op non deterministiche):
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_files):
    image = torch.load(image_files)
    return image


def load_questions(path):
    """
    Load the question file in either format:
      - JSON  : a single array of objects          -> [ {...}, {...} ]
      - JSONL : one object per line                -> {...}\n{...}
    The format is detected from the first non-whitespace character, not the extension.
    """
    path = os.path.expanduser(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Question file is empty: {path}")

    if content[0] == "[":
        questions = json.loads(content)
        if not isinstance(questions, list):
            raise ValueError(f"Expected a JSON array of objects in {path}")
        return questions

    questions = []
    for lineno, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} is not valid JSON ({e}).") from e
    return questions


def resolve_answer_key(questions, requested):
    """
    Decide which field holds the ground-truth report.
    With "auto", the first key in ANSWER_KEY_CANDIDATES present in the first sample wins.
    """
    if requested != "auto":
        return requested

    if not questions:
        return None

    first = questions[0]
    for key in ANSWER_KEY_CANDIDATES:
        if key in first:
            print(f"Ground-truth report field detected: '{key}'")
            return key

    print(
        "No ground-truth report field found "
        f"(looked for {', '.join(ANSWER_KEY_CANDIDATES)}). "
        "Outputs will carry an empty 'T-answer'."
    )
    return None


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, debug=args.debug
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1️⃣ Load the question file (JSON array or JSONL) =====
    questions = load_questions(args.question_file)
    answer_key = resolve_answer_key(questions, args.answer_key)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # ===== 2️⃣ Check and load the existing output file =====
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    processed_ids = set()
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        print(f"🔄 Existing output file detected: {answers_file}")
        with open(answers_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"✅ {len(processed_ids)} samples already processed. They will be skipped.")
        ans_file = open(answers_file, "a")  # append mode
    else:
        ans_file = open(answers_file, "w")
        print(f"🆕 Creating new output file: {answers_file}")

    # ===== 3️⃣ Iterate over the question list =====
    for line in tqdm(questions, desc="Running inference"):
        idx = line["question_id"]
        if idx in processed_ids:
            continue  # Skip already processed samples

        image_file = line["image"]
        qs = line["question"]
        metadata = line.get("metadata", "")                        # question type
        Tanswer = line.get(answer_key, "") if answer_key else ""   # ground-truth report

        cur_prompt = qs
        if model.config.mm_use_im_start_end:    # false
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        """
        A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
        Given the immunohistochemical results showing negative expression for estrogen receptor (ER), progesterone receptor (PR), and Her2, what is the molecular subtype of this breast cancer? ASSISTANT:
        """

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()       # (B, length)

        image_path = os.path.join(args.image_folder, image_file)
        image = load_image(image_path)                                  # (576, 1024)
        image_tensor = image.to(device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device), # (1, 576, 1024)
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                no_repeat_ngram_size=3,
                max_new_tokens=2048,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "image": image_file,
            "question": cur_prompt,
            "answer": outputs,
            "T-answer": Tanswer,
            "metadata": metadata
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print("✅ All samples have been processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl",
                        help="JSON array or JSONL; the format is detected automatically.")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--answer-key", type=str, default="auto",
                        help="Field holding the ground-truth report. 'auto' tries "
                             "T-answer, then text, then answer.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_seeds(seed=args.seed)

    eval_model(args)