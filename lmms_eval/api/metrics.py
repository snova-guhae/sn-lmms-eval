# the code is adapted from https://github.com/EleutherAI/lm-evaluation-harness
import math
import random
import re
import string
from collections.abc import Iterable
from typing import List

import numpy as np
import sacrebleu
import os
import time
from transformers import AutoTokenizer
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_aggregation, register_metric


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999

@register_aggregation("sum")
def mean(arr):
    return sum(arr)

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
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("binary_mean_f1")
def binary_mean_f1_score(items):
    import sklearn
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    avg_f1 = np.mean([f11, f12])
    return avg_f1


@register_aggregation("binary_f1_0")
def binary_f1_0_score(items):
    import sklearn
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    return f11


@register_aggregation("binary_f1_1")
def binary_f1_1_score(items):
    import sklearn
    golds, preds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    return f12


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


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


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


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


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
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
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
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
        base_url="https://api.sambanova.ai/v1/",
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
            if len(tokenized_messages) < 16000:
                break
            ratio = 16000 / len(tokenized_messages)
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
    output_type=["loglikelihood", "multiple_choice", "generate_until", "generate_until_multi_round"],
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
    output_type=["generate_until", "generate_until_multi_round"],
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type=["generate_until", "generate_until_multi_round"],
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type=["generate_until", "generate_until_multi_round"],
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


def stderr_for_metric(metric, bootstrap_iters: int):
    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

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


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
    assert metrics is not None, "Need to pass a list of each subtask's metric for this stderr aggregation"
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
        curr_score = ((curr_score * curr_size) + (score * size)) / (curr_size + size)  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (curr_size + size - 1) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (curr_score - score) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)
