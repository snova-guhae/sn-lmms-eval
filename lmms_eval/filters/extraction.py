import re
import sys
import unicodedata
import os
from lmms_eval.api.filter import Filter
import time
import json

class WhitespaceFilter(Filter):
    """ """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps

# Extraction prompt comes from mmlongbench-doc paper, but should be applicable to other datasets
EXTRACTION_PROMPT = """
Given the question and analysis, you are tasked to extract answers with required
formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2)
Float, (3) String and (4) List. If you find the analysis the question can not
be answered from the given documents, type "Not answerable". Exception: If the
analysis only tells you that it can not read/understand the images or documents,
type "Fail to answer".
- Please make your response as concise as possible. Also note that your response
should be formatted as below:
‘‘‘
Extracted answer: [answer]
Answer format: [answer format]
‘‘‘
Please read the following example, then extract the answer from the model response
and type it at the end of the prompt.
---
Question: List the primary questions asked about the services in this report.
Analysis: The primary questions asked about the services in the report for The Limes
Residential Home are:
1. Is the service safe?
2. Is the service effective?
3. Is the service caring?
4. Is the service responsive?
5. Is the service well-led?
Extracted answer: [’Is the servife safe?’, ’Is the service effective’, ’Is the serve
caring?’, ’Is the service responsive?’, ’Is the service well-led?’]
Answer format: List
---
Question: How many regulations of the HSCA 2008 are breached in all according to
this report?
Analysis: According to the report, the provider breached 10 Health and Social Care
Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:
1. Regulation 13: Safeguarding service users from abuse and improper treatment
2. Regulation 12: Safe care and treatment
3. Regulation 18: Staffing
4. Regulation 11: Need for consent
5. Regulation 10: Dignity and respect
6. Regulation 9: Person-centred care
7. Regulation 17: Good governance
8. Regulation 18 (CQC Registration Regulations 2009): Notification of other
incidents
9. Regulation 18: Failure to maintain an accurate and up-to-date care plan
27
10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of
practice effectively
These breaches involve issues concerning staffing, safeguarding, medicines
management, dignity and respect, consent, care planning, governance, and
failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer
---
Question: According to the survey that is the percentage of Chinese who are paying
more or about the same attention to politics after Trump’s election?
Analysis: The survey provided does not specify the percentage of Chinese individuals
specifically who are paying more or about the same attention to politics after
Trump’s election. The report focuses primarily on American demographics and
does not include specific details about the Chinese population in relation to
this question. If you need information about a different demographic or a
summary of the findings from the American demographic, I can certainly help
with that!
Extracted answer: Not answerable
Answer format: None
---
Question: How many quotations from male respondent over 50 years old are included in
this report?
Analysis: The image you’ve provided appears to be a screenshot of a document with
multiple charts. However, the text is too small and blurry to read accurately.
If you can provide a clearer image or more context, I might be able to help you
with your question.
Extracted answer: Fail to answer
Answer format: None
---
Question: {}
Analysis: {}
"""
class SambaFilter(Filter):
    formatters = {
        "Integer": int,
        "String": str,
        "Float": float,
        "List": lambda x: json.loads(x.replace("’",'"').replace("\n",'\\n')),
        "None": lambda x: None
    }
    def __init__(self):
        from openai import OpenAI

        key = os.getenv("SAMBAKEY", None)
        if key is None:
            raise ValueError("API key not found. Please set the SAMBAKEY environment variable.")
        self.client = OpenAI(
            base_url="https://api.sambanova.ai/v1/",
            api_key=key,
        )
        self.max_attempts = 5

    def apply(self, resps, docs):
        def extract_answer(response_text, doc):
            message = [
                {"role": "system", "content": "You are a highly efficient assistant. You are to be as fair and accurate"},
                {
                    "role": "user", 
                    "content": EXTRACTION_PROMPT.format(doc["question"],response_text)
                }
            ]
            for attempt in range(self.max_attempts):
                try:
                    completion = self.client.chat.completions.create(model="Meta-Llama-3.1-405B-Instruct", messages=message, max_tokens=1024, temperature=0.0, stop=["<|eot_id|>", "<|eom_id|>"])
                    response_text = completion.choices[0].message.content
                    extracted_answer = re.findall("Extracted answer: (.*)\n",response_text)[0]
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed with error: {str(e)}")
                    if attempt < self.max_attempts - 1:
                        time.sleep(30)
                    else:
                        print(f"All {self.max_attempts} attempts failed with exception {str(e)}")
                        raise e
            return extracted_answer
        answers = [extract_answer(resp, doc) for resp,doc in zip(resps,docs)]
        return answers
        

class RegexFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps


class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(f":[\s]*({without_paren_fallback_regex})")

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
                    if not match:
                        match = find_match(without_paren_fallback_regex, resp, without_paren_to_target)
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


class ExtendedRegexFilter(RegexFilter):
    punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def filter_ignores(self, st):
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)

        if self.ignore_case:
            st = st.lower()

        if self.ignore_punctuation:
            # https://stackoverflow.com/a/266162
            st = st.translate(self.punct_tbl)
        return st

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match


# Designed for the AI2D/RealworldQA dataset
class SimpleMultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
