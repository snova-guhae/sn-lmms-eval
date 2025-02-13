from pdf2image import convert_from_path
from lmms_eval.api.metrics import anls

import json

def samba_docintel_doc_to_visual(doc):
    # Don't love having to hardcode this path, but the docs aren't super accesible from the HF dataset
    images = convert_from_path(f'/import/ml-sc-scratch1/matte/samba_docintel/{doc["doc_id"]}')
    return images


def samba_docintel_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def samba_docintel_process_results(doc, results):
    pred = results[0]
    score = samba_docintel_correct(pred, doc["answer"], doc["answer_format"])
    evidence_pages = json.loads(doc["evidence_pages"])
    doc["evidence_sources"] = doc["evidence_sources"].replace("'","\"")
    evidence_types = json.loads(doc["evidence_sources"])

    # For now we are skipping Layout questions and questions that need information from multiple pages, ideally we could make the skip filter configurable.
    if len(evidence_pages) > 1 or ("Generalized-text (Layout)" in evidence_types):
        return {"skipped_questions":1}
    
    score = 1.0 if score else 0.0

    return_dict = {"overall": score}
    evidence_type = evidence_types[0] if len(evidence_types) >0 else "Unanswerable"
    evidence_type = evidence_type.replace(" ","_")
    return_dict[f"evidence_{evidence_type.lower()}"] = score
    return_dict[f"format_{doc['answer_format'].lower()}"] = score
    return return_dict

type_lookup = {
    int: "Int",
    str: "Str",
    float: "Float",
    list: "List",
    type(None): "None"
}
def samba_docintel_correct(pred, gt, answer_type):
    # Correctness logic from mmlongbench-doc paper
    if answer_type == "Int":
        try:
            return int(pred) == int(gt)
        except:
            return 0
    elif answer_type == "Str":
        anls_score = anls([gt],[pred])["anls"]
        return anls_score
    elif answer_type == "Float":
        try:
            pred_float = float(pred.replace("%",""))
            gt_float = float(gt.replace("%",""))
            delta = abs((pred_float - gt_float)/gt_float)
            return delta <= .01
        except:
            return 0
    elif answer_type == "List":
        try:
            pred_list = sorted(json.loads(pred))
            gt_list = sorted(json.loads(gt))
            if len(pred_list) != len(gt_list):
                return 0
            correctness = [samba_docintel_correct(x,y,type_lookup[x]) for x,y in zip(pred_list,gt_list)]
            return min(correctness)
        except:
            return 0
    elif answer_type == "None":
        return pred == gt