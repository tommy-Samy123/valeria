#!/usr/bin/env python3
"""Generate Valeria's synthetic text triage dataset.

Each example is a chat in the format used for training:
  system (training/config/system_prompt.txt) -> user (spoken-style injury description) -> assistant (JSON label)

Leakage control: every template, modifier and out-of-scope phrase is assigned to exactly ONE split
(train / val / test) by its position in its list, so the val/test sets use wording the model never saw.
The judged science-fair test scenarios must still be written separately by hand and never added here.

Usage:
  python3 training/scripts/build_text_dataset.py            # writes training/data/text/{train,val,test}.jsonl
  python3 training/scripts/build_text_dataset.py --seed 7 --train-per-category 400
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (ROOT / "config" / "system_prompt.txt").read_text().strip()
LABELS = json.loads((ROOT / "config" / "labels.json").read_text())
CATEGORIES = set(LABELS["categories"])
RED_FLAGS = set(LABELS["red_flags"])

# Most urgent first; decides the primary category when several injuries are named.
PRIORITY = ["laceration", "burn", "fracture", "abrasion", "sprain", "nosebleed", "bruise"]

# subject, possessive, pronoun, be, have, second-person possessive (what Valeria would say back)
SUBJECTS = [
    ("I", "my", "I", "I'm", "I have", "your"),
    ("my son", "his", "he", "he's", "he has", "his"),
    ("my daughter", "her", "she", "she's", "she has", "her"),
    ("my friend", "their", "they", "they're", "they have", "their"),
    ("my mom", "her", "she", "she's", "she has", "her"),
    ("my dad", "his", "he", "he's", "he has", "his"),
    ("my brother", "his", "he", "he's", "he has", "his"),
    ("my sister", "her", "she", "she's", "she has", "her"),
    ("my kid", "their", "they", "they're", "they have", "their"),
]
FIRST_PERSON_WEIGHT = 0.55  # most users describe their own injury

BODY = {
    "laceration": ["hand", "finger", "thumb", "palm", "arm", "forearm", "leg", "knee", "foot", "lip", "chin", "forehead"],
    "abrasion": ["knee", "elbow", "palm", "hand", "shin", "arm", "leg", "hip", "chin", "knees", "elbows"],
    "burn": ["hand", "finger", "fingers", "arm", "wrist", "forearm", "palm", "leg", "foot", "thumb"],
    "sprain": ["ankle", "wrist", "knee", "thumb", "finger"],
    "bruise": ["arm", "leg", "shin", "thigh", "knee", "hip", "shoulder", "elbow"],
    "fracture": ["arm", "wrist", "leg", "ankle", "finger", "collarbone", "toe", "shin"],
    "nosebleed": ["nose"],
    "unclear": ["arm", "leg", "hand", "foot", "back", "knee", "elbow", "wrist", "ankle", "shoulder"],
}

# (template, allowed body parts or None for the category default)
# Placeholders: {S} subject, {P} possessive, {PR} pronoun, {BE} "I'm/he's", {HAVE} "I have/he has", {B} body part
TEMPLATES: dict[str, list[tuple[str, list[str] | None]]] = {
    "laceration": [
        ("{S} cut {P} {B}", None),
        ("{S} cut {P} {B} with a knife", None),
        ("{S} sliced {P} {B} while cutting vegetables", ["finger", "thumb", "hand", "palm"]),
        ("{S} cut {P} {B} on broken glass", None),
        ("there's a cut on {P} {B} and it's bleeding", None),
        ("{BE} bleeding from a cut on {P} {B}", None),
        ("{S} cut {P} {B} on the lid of a can", ["finger", "thumb", "hand", "palm"]),
        ("{S} nicked {P} {B} shaving", ["leg", "knee", "chin", "lip"]),
        ("the knife slipped and cut {P} {B}", ["finger", "thumb", "hand", "palm"]),
        ("{HAVE} a cut on {P} {B}", None),
        ("{S} cut {P} {B} on a sharp piece of metal", None),
        ("{P} {B} is cut and bleeding", None),
        ("{S} got cut by a piece of glass on {P} {B}", None),
        ("blood is coming out of a cut on {P} {B}", None),
        ("{S} cut {P} {B} with scissors", ["finger", "thumb", "hand", "palm"]),
        ("{S} cut {P} {B} opening a box", ["finger", "thumb", "hand", "palm"]),
        ("{S} split {P} {B} open on the edge of the table", ["lip", "chin", "forehead"]),
        ("{S} sliced {P} {B} on a box cutter", ["finger", "thumb", "hand", "palm", "arm"]),
        ("{S} got a cut on {P} {B} from a broken plate", None),
        ("there's blood everywhere from a small cut on {P} {B}", ["finger", "thumb", "hand"]),
    ],
    "abrasion": [
        ("{S} scraped {P} {B}", None),
        ("{S} fell off {P} bike and scraped {P} {B}", None),
        ("{S} skinned {P} {B}", None),
        ("{S} fell on the sidewalk and {P} {B} is all scraped up", None),
        ("{S} got road rash on {P} {B}", None),
        ("{S} grazed {P} {B} on the brick wall", None),
        ("{S} tripped and scraped {P} {B} on the concrete", None),
        ("{HAVE} a scrape on {P} {B}", None),
        ("{S} slid on the gym floor and got a rug burn on {P} {B}", ["knee", "knees", "elbow", "elbows", "arm", "leg"]),
        ("{P} {B} got scratched up when {PR} fell", None),
        ("{S} fell skateboarding and scraped {P} {B}", None),
        ("the skin on {P} {B} is rubbed off", None),
        ("{S} fell on the playground and scraped {P} {B}", None),
        ("{S} scuffed {P} {B} on the pavement", None),
        ("{S} wiped out on a scooter and {P} {B} is raw and bleeding a little", None),
        ("{P} {B} has a scrape and it stings", None),
        ("{S} fell on the gravel path and {P} {B} is scraped", None),
        ("{S} got a carpet burn on {P} {B}", ["knee", "knees", "elbow", "elbows"]),
    ],
    "burn": [
        ("{S} burned {P} {B}", None),
        ("{S} burned {P} {B} on the stove", None),
        ("{S} touched a hot pan and burned {P} {B}", ["hand", "finger", "fingers", "palm", "thumb"]),
        ("{S} spilled hot coffee on {P} {B}", None),
        ("{S} got splashed with boiling water on {P} {B}", None),
        ("{S} burned {P} {B} on the oven rack", ["hand", "finger", "fingers", "arm", "wrist", "forearm", "thumb"]),
        ("{S} grabbed the curling iron and burned {P} {B}", ["hand", "finger", "fingers", "palm", "thumb"]),
        ("hot oil splashed on {P} {B}", None),
        ("{S} burned {P} {B} with steam from the kettle", None),
        ("{P} {B} is red and stinging after {PR} touched the iron", None),
        ("{S} scalded {P} {B} with hot tea", None),
        ("{S} got burned by the hot glue gun on {P} {B}", ["finger", "fingers", "thumb", "hand"]),
        ("{S} burned {P} {B} on the campfire", None),
        ("{S} touched the motorcycle exhaust pipe and burned {P} {B}", ["leg", "foot"]),
        ("{P} {B} got burned taking a tray out of the oven", None),
        ("there's a small blister on {P} {B} from a burn", None),
        ("{S} burned {P} {B} on a hot seat belt buckle", ["hand", "finger", "fingers", "leg", "thumb"]),
        ("{S} got burnt by the toaster on {P} {B}", ["hand", "finger", "fingers", "thumb"]),
    ],
    "sprain": [
        ("{S} sprained {P} {B}", None),
        ("{S} rolled {P} {B} playing basketball", ["ankle"]),
        ("{S} twisted {P} {B}", None),
        ("{S} twisted {P} {B} stepping off the curb", ["ankle", "knee"]),
        ("{P} {B} is swollen after {PR} twisted it", None),
        ("{S} landed wrong on {P} {B} and it's swollen", ["ankle", "knee", "wrist"]),
        ("{S} jammed {P} {B} playing volleyball", ["finger", "thumb"]),
        ("{S} fell and bent {P} {B} back and now it hurts to move", ["wrist", "finger", "thumb"]),
        ("{S} rolled {P} {B} on the stairs", ["ankle"]),
        ("{P} {B} hurts and is swelling up after soccer", ["ankle", "knee"]),
        ("{S} tweaked {P} {B} at practice", None),
        ("{S} turned {P} {B} running on the trail", ["ankle", "knee"]),
        ("{PR} can still walk but {P} {B} is puffy after twisting it", ["ankle", "knee"]),
        ("{S} slipped on ice and twisted {P} {B}", None),
        ("{P} {B} got twisted in a tackle and it's swelling", ["ankle", "knee", "wrist"]),
        ("{S} caught {P} {B} wrong catching a ball", ["finger", "thumb", "wrist"]),
    ],
    "bruise": [
        ("{S} bumped {P} {B} on the table", None),
        ("{HAVE} a bruise on {P} {B}", None),
        ("{S} banged {P} {B} on the door", None),
        ("{S} got hit in the {B} with a ball", None),
        ("{P} {B} is bruised", None),
        ("{S} walked into the corner of the bed and {P} {B} is purple", None),
        ("there's a big bump on {P} {B}", None),
        ("{S} got kicked in the {B} at soccer", None),
        ("{S} hit {P} {B} on the car door", None),
        ("{P} {B} has a black and blue mark", None),
        ("{S} ran into the coffee table and {P} {B} is sore and bruised", ["shin", "leg", "knee", "thigh"]),
        ("{S} got a bruise on {P} {B} from the trampoline", None),
    ],
    "fracture": [
        ("{S} broke {P} {B}", None),
        ("{S} fell and heard a crack in {P} {B}", None),
        ("{P} {B} looks broken", None),
        ("{PR} can't put any weight on {P} {B} and heard a snap", ["ankle", "leg"]),
        ("I think {P} {B} is broken", None),
        ("{S} fell off the trampoline and {P} {B} might be fractured", None),
        ("{S} fractured {P} {B}", None),
        ("{P} {B} is really painful and {PR} can't move it at all after the fall", None),
        ("{S} slammed {P} {B} in the car door and it might be broken", ["finger", "toe"]),
        ("{S} heard a pop and now {P} {B} won't move", None),
        ("{S} fell off the monkey bars and might have broken {P} {B}", ["arm", "wrist", "collarbone"]),
        ("{S} stubbed {P} {B} so hard it might be broken", ["toe"]),
    ],
    "nosebleed": [
        ("{HAVE} a nosebleed", None),
        ("{P} nose is bleeding", None),
        ("{S} got a bloody nose", None),
        ("{P} nose started bleeding out of nowhere", None),
        ("{S} got hit in the nose and it's bleeding", None),
        ("blood is dripping from {P} nose", None),
        ("{HAVE} a bloody nose that started a few minutes ago", None),
        ("{S} picked {P} nose and now it's bleeding", None),
        ("{P} nose is bleeding a little after blowing it", None),
        ("there's blood coming out of {P} nose", None),
    ],
    "unclear": [
        ("{S} hurt {P} {B}", None),
        ("{P} {B} hurts", None),
        ("something happened to {P} {B}", None),
        ("{S} injured {P} {B}", None),
        ("{S} had an accident", []),
        ("{S} fell off {P} bike", []),
        ("{S} got hurt at the park", []),
        ("{P} {B} is really sore", None),
        ("{S} fell down", []),
        ("there's something wrong with {P} {B}", None),
        ("{S} slipped in the shower", []),
        ("{S} got hurt playing football", []),
        ("{P} {B} feels weird after the fall", None),
        ("{S} crashed {P} skateboard", []),
    ],
}

# Red-flag add-ons appended to a routine injury description: (category, suffix, flags, allowed body parts)
MODIFIERS: list[tuple[str, str, list[str], list[str] | None]] = [
    ("laceration", "and it won't stop bleeding", ["severe_bleeding"], None),
    ("laceration", "and blood is spurting out", ["severe_bleeding"], None),
    ("laceration", "and the blood is soaking through the towel", ["severe_bleeding"], None),
    ("laceration", "and it's really deep, I can see fat", ["deep_wound"], None),
    ("laceration", "and the cut is gaping open", ["deep_wound"], None),
    ("laceration", "and there's a piece of glass still stuck in it", ["embedded_object"], None),
    ("laceration", "and {PR_BE_LOW} pale and feels faint", ["severe_bleeding"], None),
    ("laceration", "and it's been bleeding through the bandage for twenty minutes", ["severe_bleeding"], None),
    ("laceration", "and {PR} can't feel {P} fingers now", ["numb_or_cold_limb"], ["hand", "finger", "thumb", "palm", "arm", "forearm"]),
    ("laceration", "and the cut is really long and deep", ["deep_wound"], None),
    ("burn", "and the burn is bigger than {P} hand", ["large_burn"], ["arm", "leg", "forearm"]),
    ("burn", "and it's blistering over the whole area", ["large_burn"], ["arm", "leg", "forearm", "hand"]),
    ("burn", "and the skin looks white and leathery", ["large_burn"], None),
    ("burn", "and it covers most of {P} {B}", ["large_burn"], ["arm", "leg", "forearm", "hand"]),
    ("burn", "and some of it splashed on {P} face too", ["sensitive_area_burn"], None),
    ("burn", "and it got in {P} eye", ["sensitive_area_burn", "eye_injury"], None),
    ("burn", "and the skin is peeling off in big pieces", ["large_burn"], None),
    ("burn", "and now {PR_BE_LOW} having trouble breathing", ["breathing"], None),
    ("fracture", "and the bone is sticking out", ["bone_exposed"], None),
    ("fracture", "and it's bent the wrong way", ["deformity"], None),
    ("fracture", "and {P} foot is cold and blue", ["numb_or_cold_limb"], ["ankle", "leg", "shin"]),
    ("fracture", "and {P} {B} is bent at a weird angle", ["deformity"], None),
    ("fracture", "and you can see bone", ["bone_exposed"], None),
    ("fracture", "and {P} hand is numb and tingly", ["numb_or_cold_limb"], ["arm", "wrist", "finger"]),
    ("abrasion", "and {PR} hit {P} head and {PR_BE_LOW} confused", ["head_injury"], None),
    ("abrasion", "and {PR} hit {P} head and threw up", ["head_injury"], None),
    ("abrasion", "and {PR} blacked out for a second", ["unconscious", "head_injury"], None),
    ("abrasion", "and there's a big piece of metal stuck in it", ["embedded_object"], None),
    ("abrasion", "and {PR_BE_LOW} not making sense after hitting {P} head", ["head_injury"], None),
    ("sprain", "and {P} foot is numb and cold", ["numb_or_cold_limb"], ["ankle"]),
    ("sprain", "and {P} fingers are turning blue", ["numb_or_cold_limb"], ["wrist", "finger", "thumb"]),
    ("sprain", "and {PR} passed out from the pain", ["unconscious"], None),
    ("bruise", "and {PR} hit {P} head too and keeps throwing up", ["head_injury"], None),
    ("bruise", "and now {PR_BE_LOW} having chest pain", ["chest_pain"], None),
    ("bruise", "and {PR} got hit in the eye too and can't see well", ["eye_injury"], None),
    ("nosebleed", "and it's been bleeding for more than twenty minutes", ["severe_bleeding"], None),
    ("nosebleed", "and {PR} hit {P} head really hard and seems confused", ["head_injury"], None),
    ("nosebleed", "and {PR_BE_LOW} having trouble breathing", ["breathing"], None),
]

# Red flags with no specific supported injury -> category "none", emergency true.
# {S3} uses third-person subjects only; first-person lines are written out.
EMERGENCY_STANDALONE: list[tuple[str, list[str]]] = [
    ("{S3} is not breathing", ["breathing"]),
    ("I can't breathe", ["breathing"]),
    ("{S3} is choking", ["breathing"]),
    ("{S3} can't breathe and {PR3_POSS} lips are turning blue", ["breathing"]),
    ("{S3} passed out and won't wake up", ["unconscious"]),
    ("{S3} fainted and isn't responding", ["unconscious"]),
    ("{S3} is having a seizure", ["seizure"]),
    ("{S3} is shaking and won't respond", ["seizure", "unconscious"]),
    ("I have really bad chest pain", ["chest_pain"]),
    ("{S3} says {PR3} has chest pain and feels sweaty", ["chest_pain"]),
    ("{S3} hit {PR3_POSS} head and is throwing up", ["head_injury"]),
    ("{S3} fell and hit {PR3_POSS} head and now {PR3} is really sleepy and confused", ["head_injury"]),
    ("I hit my head and I can't remember what happened", ["head_injury"]),
    ("{S3} got electrocuted by an outlet", ["chemical_or_electrical_burn"]),
    ("{S3} spilled bleach in {PR3_POSS} eye", ["chemical_or_electrical_burn", "eye_injury"]),
    ("{S3} got something sharp stuck in {PR3_POSS} eye", ["eye_injury", "embedded_object"]),
    ("{S3} is having a hard time breathing after eating peanuts", ["breathing"]),
    ("I'm wheezing and can't catch my breath", ["breathing"]),
    ("{S3} collapsed on the floor", ["unconscious"]),
    ("{S3} got stabbed", ["deep_wound", "severe_bleeding"]),
    ("{S3} is bleeding really badly and getting pale", ["severe_bleeding"]),
    ("{S3} stepped on a nail and it's still in {PR3_POSS} foot", ["embedded_object"]),
]

OUT_OF_SCOPE = [
    "what's the weather tomorrow", "set a timer for ten minutes", "tell me a joke", "how do I make pancakes",
    "thank you", "what can you do", "I'm fine now", "never mind", "can you play some music", "what time is it",
    "I have a headache", "I feel sick to my stomach", "my tooth hurts", "I have a sore throat", "I can't sleep",
    "how many calories are in an apple", "what's the capital of France", "I'm bored", "who made you",
    "how do I tie a tie", "remind me to buy milk", "I have a stuffy nose", "my stomach hurts after lunch",
    "what should I eat for dinner", "is it going to rain", "hello", "good morning valeria", "how are you",
]

UNCLEAR_CLARIFY_BODY = [
    "What happened to {Y} {B}? Is it bleeding, burned, swollen, or bent?",
    "Is {Y} {B} bleeding, burned, or swollen?",
    "What does {Y} {B} look like: is there a cut, a scrape, a burn, or swelling?",
]
UNCLEAR_CLARIFY_NOBODY = [
    "Where does it hurt, and is anything bleeding, burned, or swollen?",
    "What part of the body is hurt, and what does it look like?",
    "Is anyone bleeding, burned, or unable to move a body part? Did anyone hit their head?",
]

PREFIXES = ["", "", "", "valeria ", "hey valeria ", "help ", "um ", "okay so ", "valeria help ", "uh "]
SUFFIXES = ["", "", "", " what do I do", " help", " what should I do", " please help", " what do I do now"]
# Realistic speech-recognition confusions (word -> misheard)
STT_SWAPS = {
    "burned": "burnt", "burn": "bern", "cut": "cup", "scraped": "scrapped", "sprained": "sprayed",
    "ankle": "angle", "wrist": "risk", "knee": "need", "bleeding": "bleating", "nose": "knows",
    "bruise": "brews", "elbow": "elbo", "finger": "fingers", "scalded": "scolded", "twisted": "twister",
    "thumb": "some", "broke": "brook",
}


def split_of(index: int) -> str:
    """Assign list positions to splits ~80/10/10. Deterministic, independent of the random seed."""
    r = index % 10
    return "val" if r == 7 else "test" if r == 3 else "train"


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def subject(self, third_person_only: bool = False):
        if third_person_only:
            return self.rng.choice(SUBJECTS[1:])
        if self.rng.random() < FIRST_PERSON_WEIGHT:
            return SUBJECTS[0]
        return self.rng.choice(SUBJECTS[1:])

    def fill(self, template: str, subj, body: str | None) -> str:
        s, p, pr, be, have, _ = subj
        text = template.format(
            S=s, P=p, PR=pr, BE=be, HAVE=have, B=body or "",
            PR_BE_LOW=be, S3=s, PR3=pr, PR3_POSS=p,
        )
        return " ".join(text.split())

    def speechify(self, text: str, noise: float) -> str:
        words = text.split()
        if self.rng.random() < noise and len(words) > 4:
            # Misrecognise one word, or drop a filler word.
            idxs = [i for i, w in enumerate(words) if w.lower() in STT_SWAPS]
            if idxs and self.rng.random() < 0.6:
                i = self.rng.choice(idxs)
                words[i] = STT_SWAPS[words[i].lower()]
            else:
                droppable = [i for i, w in enumerate(words) if w.lower() in {"the", "a", "and", "my", "it's", "is", "on"}]
                if droppable:
                    del words[self.rng.choice(droppable)]
        text = " ".join(words)
        text = self.rng.choice(PREFIXES) + text + self.rng.choice(SUFFIXES)
        style = self.rng.random()
        if style < 0.55:
            text = text.lower()
        elif style < 0.85:
            text = text[0].upper() + text[1:]
        else:
            text = text[0].upper() + text[1:] + self.rng.choice([".", "!", "?"])
        return text


def target(category, also=(), red_flags=(), body_part=None, clarify=None) -> dict:
    assert category in CATEGORIES, category
    assert all(f in RED_FLAGS for f in red_flags), red_flags
    return {
        "category": category,
        "also": list(also),
        "emergency": bool(red_flags),
        "red_flags": list(red_flags),
        "body_part": body_part,
        "clarify": clarify,
    }


def example(user_text: str, label: dict, kind: str) -> dict:
    return {
        "kind": kind,
        "label": label,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(label, separators=(",", ":"))}]},
        ],
    }


def templates_for(split: str, items: list):
    return [t for i, t in enumerate(items) if split_of(i) == split]


def build(split: str, per_category: int, g: Gen, noise: float) -> list[dict]:
    out: list[dict] = []
    scale = per_category

    # 1) Routine single injuries.
    for cat in PRIORITY:
        temps = templates_for(split, TEMPLATES[cat])
        for _ in range(scale):
            tpl, parts = g.rng.choice(temps)
            body = g.rng.choice(parts or BODY[cat])
            subj = g.subject()
            text = g.fill(tpl, subj, body)
            out.append(example(g.speechify(text, noise), target(cat, body_part=body), "single"))

    # 2) Unclear -> clarifying question.
    temps = templates_for(split, TEMPLATES["unclear"])
    for _ in range(int(scale * 0.7)):
        tpl, parts = g.rng.choice(temps)
        subj = g.subject()
        if parts == []:
            body, q = None, g.rng.choice(UNCLEAR_CLARIFY_NOBODY)
        else:
            body = g.rng.choice(parts or BODY["unclear"])
            q = g.rng.choice(UNCLEAR_CLARIFY_BODY).format(Y=subj[5], B=body)
        text = g.fill(tpl, subj, body)
        out.append(example(g.speechify(text, noise), target("unclear", body_part=body, clarify=q), "unclear"))

    # 3) Routine injury + red-flag modifier -> emergency.
    mods = templates_for(split, MODIFIERS)
    for _ in range(int(scale * 1.4)):
        cat, suffix, flags, parts = g.rng.choice(mods)
        tpl, tparts = g.rng.choice(templates_for(split, TEMPLATES[cat]) or TEMPLATES[cat])
        allowed = [b for b in (tparts or BODY[cat]) if parts is None or b in parts] or (parts or BODY[cat])
        body = g.rng.choice(allowed)
        subj = g.subject()
        text = g.fill(tpl + " " + suffix, subj, body)
        out.append(example(g.speechify(text, noise * 0.5), target(cat, red_flags=flags, body_part=body), "red_flag"))

    # 4) Stand-alone emergencies.
    em = templates_for(split, EMERGENCY_STANDALONE)
    for _ in range(int(scale * 0.7)):
        tpl, flags = g.rng.choice(em)
        subj = g.subject(third_person_only=True)
        text = g.fill(tpl, subj, None)
        out.append(example(g.speechify(text, noise * 0.5), target("none", red_flags=flags), "emergency"))

    # 5) Out of scope.
    oos = templates_for(split, OUT_OF_SCOPE)
    for _ in range(int(scale * 0.5)):
        out.append(example(g.speechify(g.rng.choice(oos), noise), target("none"), "out_of_scope"))

    # 6) Two injuries in one sentence.
    for _ in range(int(scale * 0.7)):
        a, b = g.rng.sample(PRIORITY, 2)
        first, second = sorted([a, b], key=PRIORITY.index)
        subj = g.subject()
        parts_text, bodies = [], {}
        pronoun_subj = (subj[2],) + subj[1:]  # "my sister ... and she ..."
        for n, cat in enumerate((a, b)):
            tpl, parts = g.rng.choice(templates_for(split, TEMPLATES[cat]))
            body = g.rng.choice(parts or BODY[cat])
            bodies[cat] = body
            parts_text.append(g.fill(tpl, subj if n == 0 else pronoun_subj, body))
        text = " and ".join(parts_text)
        out.append(example(g.speechify(text, noise * 0.5), target(first, also=[second], body_part=bodies[first]), "multi"))

    # De-duplicate identical user texts within the split.
    seen, uniq = set(), []
    for ex in out:
        key = ex["messages"][1]["content"][0]["text"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(ex)
    g.rng.shuffle(uniq)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--train-per-category", type=int, default=350)
    ap.add_argument("--eval-per-category", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.25, help="probability of a simulated STT error per example")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "text")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    g = Gen(args.seed)
    all_texts: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        n = args.train_per_category if split == "train" else args.eval_per_category
        rows = build(split, n, g, args.noise)
        all_texts[split] = {r["messages"][1]["content"][0]["text"].lower() for r in rows}
        with open(args.out / f"{split}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                r["id"] = f"text-{split}-{i:05d}"
                f.write(json.dumps(r) + "\n")
        kinds = Counter(r["kind"] for r in rows)
        cats = Counter(r["label"]["category"] for r in rows)
        print(f"{split}: {len(rows)} examples | kinds {dict(kinds)} | categories {dict(cats)}")

    overlap = (all_texts["train"] & all_texts["test"]) | (all_texts["train"] & all_texts["val"])
    print(f"exact text overlap between train and val/test: {len(overlap)}")


if __name__ == "__main__":
    main()
