import re
import spacy
from collections.abc import Iterable
from typing import Any

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    # fall back to invoking spaCy's CLI via subprocess to download the model
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
OBJECTS = {"dobj", "obj", "dative", "attr", "oprd", "pobj"}
BREAKER_POS = {"CCONJ", "VERB"}
NEGATIONS = {"no", "not", "n't", "never", "none"}


# extract q&a blocks from deposition transcipts
def extract_qa_blocks(transcript: str) -> list[dict]:
    """
    Extracts Q&A blocks from a deposition transcript.
    Returns a list of dictionaries with 'question' and 'answer' keys.
    """
    pattern = re.compile(r'Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Q:|\Z)', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(transcript)
    
    blocks = []
    for q, a in matches:
        blocks.append({
            'question': q.strip(),
            'answer': a.strip()
        })
        
    return blocks


def contains_conj(dep_set):
    return (
        "and" in dep_set
        or "or" in dep_set
        or "nor" in dep_set
        or "but" in dep_set
        or "yet" in dep_set
        or "so" in dep_set
        or "for" in dep_set
    )


def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        rights = list(sub.rights)
        right_deps = {tok.lower_ for tok in rights}
        if contains_conj(right_deps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if more_subs:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        rights = list(obj.rights)
        right_deps = {tok.lower_ for tok in rights}
        if contains_conj(right_deps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if more_objs:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


def _find_subs(tok):
    head = tok.head
    while head.pos_ not in {"VERB", "AUX", "NOUN"} and head.head != head:
        head = head.head

    if head.pos_ in {"VERB", "AUX"}:
        subs = [left for left in head.lefts if left.dep_ in SUBJECTS]
        if subs:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        if head.head != head:
            return _find_subs(head)

    if head.pos_ == "NOUN":
        return [head], _is_negated(tok)

    return [], False


def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS and dep.dep_ != "intj":
            return True

    if tok.dep_ in {"ccomp", "xcomp"} and tok.head is not tok:
        return _is_negated(tok.head)

    return False


def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for verb in verbs:
        subs, verb_negated = _get_all_subs(verb)
        for sub in subs:
            svs.append((sub.orth_, "!" + verb.orth_ if verb_negated else verb.orth_))
    return svs


def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend(
                [
                    tok
                    for tok in dep.rights
                    if tok.dep_ in OBJECTS
                    or (tok.pos_ == "PRON" and tok.lower_ == "me")
                    or (is_pas and tok.dep_ == "pobj")
                ]
            )
    return objs


def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            for verb in verbs:
                rights = list(verb.rights)
                objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                objs.extend(_get_objs_from_prepositions(rights, is_pas))
                if objs:
                    return verb, objs
    return None, None


def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            verb = dep
            rights = list(verb.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if objs:
                return verb, objs
    return None, None


def _get_all_subs(verb):
    verb_negated = _is_negated(verb)
    subs = [tok for tok in verb.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if subs:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        found_subs, verb_negated = _find_subs(verb)
        subs.extend(found_subs)
    return subs, verb_negated


def _find_verbs(tokens):
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if not verbs:
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs


def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and tok.dep_ not in {"aux", "auxpass"}


def _is_verb(tok):
    return tok.pos_ in {"VERB", "AUX"}


def _right_of_verb_is_conj_verb(verb):
    rights = list(verb.rights)
    if len(rights) > 1 and rights[0].pos_ == "CCONJ":
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok
    return False, verb


def _get_all_objs(verb, is_pas):
    rights = list(verb.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == "pobj")]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    if not objs and verb.lemma_ == "be":
        objs.extend([tok for tok in rights if tok.dep_ in {"acomp", "advmod"}])

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs:
        objs.extend(potential_new_objs)
        verb = potential_new_verb

    if objs:
        objs.extend(_get_objs_from_conjunctions(objs))

    return verb, objs


def _is_passive(tokens):
    return any(tok.dep_ == "auxpass" for tok in tokens)


def _get_that_resolution(toks):
    for tok in toks:
        if "that" in [left.orth_ for left in tok.lefts]:
            return tok.head
    return None


def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


def expand(item, tokens, visited):
    if item.lower_ == "that":
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, "lefts"):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if part.lower_ not in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, "rights"):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if part.lower_ not in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], "rights"):
        for item2 in parts[-1].rights:
            if item2.pos_ in {"DET", "NOUN", "PROPN", "ADJ"} and item2.i not in visited:
                visited.add(item2.i)
                parts.extend(expand(item2, tokens, visited))
            break

    return parts


def to_str(tokens):
    if isinstance(tokens, Iterable):
        return " ".join([item.text for item in tokens])
    return ""


def findSVOs(tokens):
    svos = []
    is_pas = _is_passive(tokens)
    verbs = _find_verbs(tokens)

    for verb in verbs:
        subs, verb_negated = _get_all_subs(verb)
        if not subs:
            continue

        is_conj_verb, conj_verb = _right_of_verb_is_conj_verb(verb)
        if is_conj_verb:
            v2, objs = _get_all_objs(conj_verb, is_pas)
            for sub in subs:
                for obj in objs:
                    obj_negated = _is_negated(obj)
                    if is_pas:
                        svos.append((to_str(expand(obj, tokens, set())), "!" + verb.lemma_ if verb_negated or obj_negated else verb.lemma_, to_str(expand(sub, tokens, set()))))
                        svos.append((to_str(expand(obj, tokens, set())), "!" + v2.lemma_ if verb_negated or obj_negated else v2.lemma_, to_str(expand(sub, tokens, set()))))
                    else:
                        svos.append((to_str(expand(sub, tokens, set())), "!" + verb.lemma_ if verb_negated or obj_negated else verb.lemma_, to_str(expand(obj, tokens, set()))))
                        svos.append((to_str(expand(sub, tokens, set())), "!" + v2.lemma_ if verb_negated or obj_negated else v2.lemma_, to_str(expand(obj, tokens, set()))))
            continue

        verb, objs = _get_all_objs(verb, is_pas)
        for sub in subs:
            if objs:
                for obj in objs:
                    obj_negated = _is_negated(obj)
                    if is_pas:
                        svos.append((to_str(expand(obj, tokens, set())), "!" + verb.lemma_ if verb_negated or obj_negated else verb.lemma_, to_str(expand(sub, tokens, set()))))
                    else:
                        svos.append((to_str(expand(sub, tokens, set())), "!" + verb.lemma_ if verb_negated or obj_negated else verb.lemma_, to_str(expand(obj, tokens, set()))))
            else:
                svos.append((to_str(expand(sub, tokens, set())), "!" + verb.lemma_ if verb_negated else verb.lemma_))

    return svos


def _question_context(question: str) -> dict[str, str]:
    doc = nlp(question)
    context = {}
    lowered = question.lower()

    if lowered.startswith("where"):
        context["asked_about"] = "location"
    elif lowered.startswith("when") or "what time" in lowered:
        context["asked_about"] = "time"
    elif lowered.startswith("who"):
        context["asked_about"] = "person"
    elif lowered.startswith("what"):
        context["asked_about"] = "event"
    else:
        context["asked_about"] = "unknown"

    times = [_clean_context_value(ent.text) for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]
    if times:
        context["time"] = " ".join(times)

    return context


def _clean_context_value(value: str) -> str:
    return re.sub(r"\s+(and|or)$", "", value.strip(), flags=re.IGNORECASE)


def _time_context(question: str, answer: str) -> str | None:
    doc = nlp(f"{question} {answer}")
    times = [_clean_context_value(ent.text) for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]
    if times:
        return " ".join(dict.fromkeys(times))

    lowered = f"{question} {answer}".lower()
    for pattern in [
        r"\ball evening\b",
        r"\ball night\b",
        r"\baround\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        r"\bmidnight\b",
        r"\bnovember\s+\d{1,2}(?:st|nd|rd|th)?\b",
    ]:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)

    return None


def _location_context(answer: str, arg1: str) -> str | None:
    arg1_lower = arg1.lower()
    if any(word in arg1_lower for word in ["home", "warehouse", "street", "parking lot", "portland"]):
        return arg1

    doc = nlp(answer)
    locations = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}]
    return locations[0] if locations else None


def _normalize_svo(svo: tuple, question: str, sentence_text: str) -> dict[str, Any] | None:
    arg0 = svo[0]
    raw_verb = svo[1]
    arg1 = svo[2] if len(svo) > 2 else None
    negated = raw_verb.startswith("!")
    verb = raw_verb[1:] if negated else raw_verb

    if not arg0 or not verb:
        return None

    if verb == "face" and arg1 == "face":
        return None

    location = _location_context(sentence_text, arg1 or "")

    if verb in {"be", "was", "were", "is", "are"} and arg1:
        sentence_lower = sentence_text.lower()
        arg1_lower = arg1.lower()
        if location or re.search(rf"\b(at|in|on)\s+{re.escape(arg1_lower)}\b", sentence_lower):
            verb = f"{verb} at"

    return {
        "ARG0": arg0,
        "V": verb,
        "ARG1": arg1,
        "AM-TMP": _time_context(question, sentence_text),
        "AM-LOC": location,
        "negated": negated,
        "question_context": _question_context(question),
        "original_text": sentence_text,
    }


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for claim in claims:
        key = (
            claim.get("ARG0"),
            claim.get("V"),
            claim.get("ARG1"),
            claim.get("negated"),
            claim.get("original_text"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)

    return deduped


# extract arg, v/r, arg 
def extract_arguments(text: dict):
    question = text["question"]
    answer = text["answer"]
    claims = []

    for sentence in nlp(answer).sents:
        sentence_text = sentence.text.strip()
        if not sentence_text:
            continue

        for svo in findSVOs(sentence):
            claim = _normalize_svo(svo, question, sentence_text)
            if claim is not None:
                claims.append(claim)

    return _dedupe_claims(claims)


def extract_argument(text: dict):
    return extract_arguments(text)


def embed_blocks_to_db():
    # extract claims from all blocks
    pass

# calculate contradiction score for each pair of claims
def calculate_contradiction_score(): pass


if __name__ == '__main__':
    # Test the extraction
    dep1 = """
Deposition of Marcus Webb - March 14, 2023

Q: Where were you on the evening of November 3rd?
A: I was at home all evening. I ordered pizza around 7pm and watched TV.

Q: Did you speak to anyone that night?
A: No, I was alone. My wife was visiting her sister in Portland.

Q: What time did you go to sleep?
A: Around 10, maybe 10:30. I had work the next morning.

Q: Have you ever been to the Hargrove Street warehouse?
A: No, never. I don't even know where that is.

Q: Do you own a grey Honda Civic?
A: I did at the time, yes. I sold it in January.

Q: Had you met Daniel Cho before November 3rd?
A: No. I'd never heard of him before this whole thing started.
    """
    dep2 = """ 
Deposition of Marcus Webb - September 9, 2023

Q: Walk me through the evening of November 3rd again.
A: I was home. I think I went out briefly to get some groceries, maybe around 7:30, but came right back.

Q: You mentioned last time you ordered pizza. Now you're saying groceries?
A: I might have done both. I don't remember exactly, it was almost a year ago.

Q: Did anyone see you that evening?
A: My neighbor, Tom, might have seen me. We waved or something in the parking lot.

Q: What time did you go to sleep?
A: It was late. Midnight maybe. I had trouble sleeping.

Q: Had you ever visited the Hargrove Street area?
A: I mean, I've driven through that part of town. I didn't say I'd never been in that general area.

Q: And Daniel Cho - did you know him?
A: I knew of him. We had mutual friends. I don't think I'd met him face to face. """
    
    for deposition_id, transcript in [("dep1", dep1), ("dep2", dep2)]:
        print(f"\n{deposition_id} extracted claims")
        for block_index, block in enumerate(extract_qa_blocks(transcript), start=1):
            claims = extract_arguments(block)
            print(f"\nQ{block_index}: {block['question']}")
            for claim in claims:
                print(claim)
    
