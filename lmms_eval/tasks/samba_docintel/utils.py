from pdf2image import convert_from_path
from lmms_eval.api.metrics import anls
import os
import json
import re
import time
DOCUMENT_FOLDER = "/import/ml-sc-scratch1/matte/samba_docintel/documents/"

def samba_docintel_doc_to_visual(doc):
    # Don't love having to hardcode this path, but the docs aren't super accesible from the HF dataset
    images = convert_from_path(f'{DOCUMENT_FOLDER}{doc["doc_id"]}')
    return images


def samba_docintel_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def samba_docintel_doc_to_target_retrieval(doc):
    return f"Expected Answer: {doc['answer']}, Expected Pages: {doc['evidence_pages']}, Expected Types: {doc['evidence_sources']}"

def samba_docintel_process_results(doc, results):
    pred = results[0]
    score = samba_docintel_correct(pred, doc["answer"], doc["answer_format"])
    evidence_pages = json.loads(doc["evidence_pages"])
    doc["evidence_sources"] = doc["evidence_sources"].replace("'", '"')
    evidence_types = json.loads(doc["evidence_sources"])

    # For now we are skipping Layout questions and questions that need information from multiple pages, ideally we could make the skip filter configurable.
    if len(evidence_pages) > 1 or ("Generalized-text (Layout)" in evidence_types):
        return {"skipped_questions": 1}

    score = 1.0 if score else 0.0

    return_dict = {"overall": score}
    evidence_type = evidence_types[0] if len(evidence_types) > 0 else "Unanswerable"
    evidence_type = evidence_type.replace(" ", "_")
    return_dict[f"evidence_{evidence_type.lower()}"] = score
    answer_format = doc['answer_format'] if doc['answer_format'] is not None else "None"
    return_dict[f"format_{answer_format.lower()}"] = score
    return return_dict

def samba_docintel_retrieval_process_results(doc, results):

    NUM_SECONDS_TO_SLEEP = 30
    from openai import OpenAI

    key = os.getenv("SAMBAKEY", os.getenv("SAMBANOVA_API_KEY",None))
    if key is None:
        raise ValueError("API key not found. Please set the SAMBAKEY or SAMBANOVA_API_KEY environment variable.")
    client = OpenAI(
        base_url="https://api.sambanova.ai/v1/",
        api_key=key,
    )

    expected_answer = samba_docintel_doc_to_target_retrieval(doc)
    messages = []
    messages.append({"role": "system", "content": "You are a highly efficient assistant. You are to be as fair and accurate"})
    messages.append(
        {
            "role": "user",
            "content": ("I am going to give you the model's response to a question that involves a retrieval component, the response should mention the page numbers and types of evidence it used to answer the question. I will also give you the expected answer, page numbers, and evidence types for the question. I have 4 criteria I need you to score on:\n" 
            "- Whether or not the expected page numbers appear to be the main sources cited.\n" 
            "- Whether or not the expected page numbers are mentioned at all\n" 
            "- Whether or not the expected evidence types appear to be the main sources cited.\n" 
            "- Whether or not the expected evidence types are mentioned at all\n" 
            "Answer in the form of a list with an entry for each criteria in order, 1 if it meets the criteria, 0 if not. For example, if all criteria is met, return [1,1,1,1] or if the page number criterias are met, but not the evidence types, return [1,1,0,0]. Then give me an explanation of your judgement.\n"
            f"\n Here is the expected evidence to answer the question: \n\n {expected_answer} \n\n Here is the model completion: \n\n {results[0]} \n\n Judgement:"),
        }
    )
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(model="Meta-Llama-3.1-405B-Instruct", messages=messages, max_tokens=1024, temperature=0.0, stop=["<|eot_id|>", "<|eom_id|>"])
            response_text = completion.choices[0].message.content
            extracted_scores = re.findall(r"(\[\d, ?\d, ?\d, ?\d ?\])",response_text)
            if extracted_scores is None:
                print(f"Attempt failed with text: {response_text}.")
                scores = [0,0,0,0]
                break
            scores = json.loads(extracted_scores[0])
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                raise e
    page_main, page_mentioned, evidence_main, evidence_mentioned = scores
    overall = (page_main + page_mentioned + evidence_main + evidence_mentioned)/4
    return {"overall":overall, "page_main": page_main, "page_mentioned":page_mentioned,"evidence_main": evidence_main, "evidence_mentioned": evidence_mentioned, "samba_for_log": response_text}


type_lookup = {int: "Int", str: "Str", float: "Float", list: "List", type(None): "None"}


def samba_docintel_correct(pred, gt, answer_type):
    # Correctness logic from mmlongbench-doc paper
    if answer_type == "Int":
        try:
            return int(pred) == int(gt)
        except:
            return 0
    elif answer_type == "Str":
        anls_score = anls([gt], [pred])["anls"]
        return anls_score
    elif answer_type == "Float":
        try:
            pred_float = float(pred.replace("%", ""))
            gt_float = float(gt.replace("%", ""))
            delta = abs((pred_float - gt_float) / gt_float)
            return delta <= 0.01
        except:
            return 0
    elif answer_type == "List":
        try:
            pred_list = sorted(json.loads(pred))
            gt_list = sorted(json.loads(gt))
            if len(pred_list) != len(gt_list):
                return 0
            correctness = [samba_docintel_correct(x, y, type_lookup[x]) for x, y in zip(pred_list, gt_list)]
            return min(correctness)
        except:
            return 0
    elif answer_type.lower() == "none":
        return pred == gt
