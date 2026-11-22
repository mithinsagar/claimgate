import random

from claimgate.baselines import overlap_ratio
from claimgate.data import Example
from claimgate.stress import _condition, _negation, _quantity_edits, generate


def make_example(sentence: str, context: str) -> Example:
    return Example(uid="u1", split="t", question="q", context=context.lower(), context_cased=context, answer=sentence, sentences=[sentence],
                   sentence_labels=["Supported"], label=0, meta={"evidence": context})


def test_quantity_edits_have_expected_labels():
    edits = {k: (c, lab) for k, c, lab, _ in _quantity_edits("Tighten the lug nuts to 130 ft-lb.", "tighten lug nuts to 130 ft-lb .", random.Random(0))}
    assert set(edits) == {"value_shift", "unit_swap", "converted"}
    assert edits["unit_swap"] == ("Tighten the lug nuts to 130 N·m.", 1)
    assert edits["converted"][1] == 0 and "176 N·m" in edits["converted"][0]
    assert edits["value_shift"][1] == 1 and "130 ft-lb" not in edits["value_shift"][0]


def test_negation_and_condition_edits():
    assert _negation("The light will flash twice.")[0] == "The light will not flash twice."
    assert _negation("Do not open the cap.")[0] == "Do open the cap."
    assert _condition("Turn the knob clockwise before starting.")[0] == "Turn the knob clockwise after starting."


def test_generate_is_deterministic():
    ex = make_example("You should inflate the spare to 60 psi before you drive.", "You should inflate the spare to 60 psi before you drive.")
    a = [(i.kind, i.claim) for i, _ in generate([ex], seed=3)]
    b = [(i.kind, i.claim) for i, _ in generate([ex], seed=3)]
    assert a == b and {k for k, _ in a} >= {"original", "value_shift", "unit_swap", "converted", "condition_swap", "negation"}


def test_overlap_ratio_counts_long_phrases_only():
    a = "check the tire pressure once a month when cold".split()
    assert overlap_ratio(a, a) == 1.0
    # shorter sentence has 6 tokens -> phrases must be longer than min(4, 1.8) tokens; single shared words do not count
    assert overlap_ratio(a, "pressure gauges are sold separately cold".split()) == 0.0
    assert overlap_ratio(a, "always check the tire pressure first".split()) == 4 / 6


def test_ragtruth_sentence_offsets_match_text():
    from claimgate.ragtruth import _sentences_with_offsets
    text = "Mechanics earn about $20 an hour. Pay varies by state!\nSome are paid per job."
    parts = _sentences_with_offsets(text)
    assert [p[0] for p in parts] == ["Mechanics earn about $20 an hour.", "Pay varies by state!", "Some are paid per job."]
    assert all(text[s:e].strip() == t for t, s, e in parts)


def test_converted_edit_replaces_a_dual_unit_phrase_whole():
    edits = {k: c for k, c, _, _ in _quantity_edits("Service within 500 miles (805 km) of the warning.", "service within 500 miles ( 805 km ) .",
                                                    random.Random(0))}
    assert edits["converted"] == "Service within 805 km of the warning."
