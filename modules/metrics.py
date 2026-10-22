from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics import accuracy_score, recall_score, f1_score

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_scores_classification(gts, res):
    """
    gts: list or array of ground-truth labels
    res: list or array of predicted labels
    """

    eval_res = {}

    eval_res["accuracy"] = accuracy_score(gts, res)

    eval_res["recall_macro"] = recall_score(
        gts, res, labels=sorted(set(gts)), average="macro", zero_division=0
    )
    eval_res["recall_micro"] = recall_score(
        gts, res, average="micro", zero_division=0
    )
    eval_res["recall_weighted"] = recall_score(
        gts, res, average="weighted", zero_division=0
    )

    eval_res["f1_macro"] = f1_score(
        gts, res, labels=sorted(set(gts)), average="macro", zero_division=0
    )
    eval_res["f1_micro"] = f1_score(
        gts, res, average="micro", zero_division=0
    )
    eval_res["f1_weighted"] = f1_score(
        gts, res, average="weighted", zero_division=0
    )

    return eval_res
