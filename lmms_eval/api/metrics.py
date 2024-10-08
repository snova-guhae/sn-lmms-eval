import math
from collections.abc import Iterable

import numpy as np
import sacrebleu
import sklearn.metrics
import random
import evaluate
import torch
import copy
import re
from lmms_eval.api.registry import register_metric, register_aggregation
from loguru import logger as eval_logger
import os
import time
from transformers import AutoTokenizer


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    # return math.exp(-mean(items))
    items = torch.exp(torch.tensor(items)).tolist()
    return sum(items) / len(items)


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)
    return np.max(fscore)


@register_aggregation("binary_mean_f1")
def binary_mean_f1_score(items):
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    avg_f1 = np.mean([f11, f12])
    return avg_f1


@register_aggregation("binary_f1_0")
def binary_f1_0_score(items):
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    return f11


@register_aggregation("binary_f1_1")
def binary_f1_1_score(items):
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    return f12


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    # print(preds)
    return sklearn.metrics.matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


exact_match = evaluate.load("exact_match")


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match.compute(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


@register_metric(
    metric="anls",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def anls(
    references,
    predictions,
    thresh_hold=0.5,
):  # This is a passthrough function
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(predictions[0].strip().lower().split())

        # dist = levenshtein_distance(answer.lower(), detObject['answer'].lower())
        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(predictions[0].upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return {"anls": question_result}


@register_metric(metric="gpt4judge", higher_is_better=True, output_type="generate_until", aggregation="mean", query="false")
def gpt4judge(references, predictions, query):  # This is a passthrough function
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    from openai import AzureOpenAI, BadRequestError

    NUM_SECONDS_TO_SLEEP = 30
    responses = []
    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = answer
        det_answer = predictions[0]
        api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        client = AzureOpenAI(api_key=api_key, api_version="2024-02-15-preview", azure_endpoint=azure_endpoint)
        messages = []
        messages.append({"role": "system", "content": "You are a highly efficient assistant. You are to be as fair and accurate"})
        messages.append(
            {
                "role": "user",
                "content": "I am going to give you a question, the answer to the question, and model's answer to the question. You are to tell me if the model is correct. Respond [[1]] if correct and [[0]] if incorrect. Then give me an explanation of your judgement. Here is the question: \n\n What is name of university? \n\n Here is the answer to the question: \n\n University of California, San Diego \n\n Here is the model completion: \n\n UCSD \n\n Judgement:",
            }
        )
        messages.append({"role": "assistant", "content": "The answer is correct, so I rate [[1]]. \n\n Explanation: UCSD is an appropriate abbreviation for the University of California, San Diego. "})
        messages.append(
            {
                "role": "user",
                "content": f"I am going to give you a question, the answer to the question, and model's answer to the question. You are to tell me if the model is correct. Respond [[1]] if correct and [[0]] if incorrect. Then give me an explanation of your judgement. Here is the question: \n\n {query} \n\n Here is the answer to the question: \n\n { gt_answer} \n\n Here is the model completion: \n\n {det_answer} \n\n Judgement:",
            }
        )
        for attempt in range(5):
            try:
                completion = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=messages)
                response = completion.choices[0].message.content
                break  # If successful, break out of the loop

            except Exception as e:
                try:
                    error_msg = response.json()
                except:
                    error_msg = ""

                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nError message: {error_msg}")
                if attempt <= 3:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    if isinstance(e, BadRequestError) and e.code == "content_filter":
                        eval_logger.error(f"Ran into a content filter error.\n***Original question***:\n{query}\n***Original ground truth answer***:\n{gt_answer}\n***Original model response***:\n{det_answer}")
                        raise e
                    eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {str(error_msg)}")
                    response = ""

        score = int(extract_number_from_brackets(response))
        responses.append(response)
        values.append(score)

    return {"gpt4judge": max(values), "gpt4_for_log": responses}


@register_metric(metric="sambajudge", higher_is_better=True, output_type="generate_until", aggregation="mean", query="false")
def sambajudge(references, predictions, query):  # This is a passthrough function
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    responses = []
    import requests
    import json

    NUM_SECONDS_TO_SLEEP = 30
    from openai import OpenAI

    key = os.getenv("SAMBAKEY", None)
    if key is None:
        raise ValueError("API key not found. Please set the SAMBAKEY environment variable.")
    client = OpenAI(
        base_url="https://fast-api.snova.ai/v1/",
        api_key=key,
    )

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = answer
        det_answer = predictions[0]

        def create_messages(query, gt_answer, det_answer):
            messages = []
            messages.append({"role": "system", "content": "You are a highly efficient assistant. You are to be as fair and accurate"})
            messages.append(
                {
                    "role": "user",
                    "content": "I am going to give you a question, the answer to the question, and model's answer to the question. You are to tell me if the model is correct. Respond [[1]] if correct and [[0]] if incorrect. Then give me an explanation of your judgement. Here is the question: \n\n What is name of the university in San Diego? \n\n Here is the answer to the question: \n\n University of California, San Diego \n\n Here is the model completion: \n\n UCSD \n\n Judgement:",
                }
            )
            messages.append({"role": "assistant", "content": "The answer is correct, so I rate [[1]]. \n\n Explanation: UCSD is an appropriate abbreviation for the University of California, San Diego. "})
            messages.append(
                {
                    "role": "user",
                    "content": f"I am going to give you a question, the answer to the question, and model's answer to the question. You are to tell me if the model is correct. Respond [[1]] if correct and [[0]] if incorrect. Then give me an explanation of your judgement. Here is the question: \n\n  What is the capital of France? \n\n Here is the answer to the question: \n\n Paris, France \n\n Here is the model completion: \n\n Monaco \n\n Judgement:",
                }
            )
            messages.append({"role": "assistant", "content": "The answer is incorrect, so I rate [[0]]. \n\n Explanation: The model answers Monaco, but Paris is the capital of France."})
            messages.append(
                {
                    "role": "user",
                    "content": f"I am going to give you a question, the answer to the question, and model's answer to the question. You are to tell me if the model is correct. Respond [[1]] if correct and [[0]] if incorrect. Then give me an explanation of your judgement. Here is the question: \n\n {query} \n\n Here is the answer to the question: \n\n {gt_answer} \n\n Here is the model completion: \n\n {det_answer} \n\n Judgement:",
                }
            )
            return messages

        tokenizer = AutoTokenizer.from_pretrained("/import/snvm-sc-podscratch4/jonathanl/generic_checkpoints/llama_3/Meta-Llama-3-8B-Instruct")
        while True:
            messages = create_messages(query, gt_answer, det_answer)
            tokenized_messages = tokenizer.apply_chat_template(messages)
            if len(tokenized_messages) < 3600:
                break
            ratio = 4000 / len(tokenized_messages)
            ratio = min(ratio, 0.95)
            query = query[-int(ratio * len(query)) :]
            print("lessening")

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(model="Meta-Llama-3.1-405B-Instruct", messages=messages, max_tokens=1024, temperature=0.0, stop=["<|eot_id|>", "<|eom_id|>"])
                response_text = completion.choices[0].message.content
                extracted_number = extract_number_from_brackets(response_text)
                if extracted_number is None:
                    print(f"Attempt failed with text: {response_text}.")
                    score = 0.0
                    break
                score = int(extracted_number)
                break
            except Exception as e:
                eval_logger.info(f"Attempt {attempt+1} failed with error: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:
                    eval_logger.error(f"All {max_attempts} attempts failed with exception {str(e)}")
                    raise e
        responses.append(response_text)
        values.append(score)
    return {"sambajudge": max(values), "samba_for_log": responses}


def extract_number_from_brackets(string):
    # Regular expression to find numbers inside double brackets
    match = re.search(r"\[\[(\d+)\]\]", string)
    if match:
        return int(match.group(1))
    else:
        return None  # Return None if no match is found


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return items


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="binary_mean_f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="binary_mean_f1",
)
def mean_f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1_0",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="binary_f1_0",
)
def f1_0_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1_1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="binary_f1_1",
)
def f1_1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)
