import pytest

from claimgate.quantities import check, extract


def statuses(claim, evidence):
    return [v.status for v in check(claim, evidence)]


def test_extracts_numbers_units_ranges():
    qs = extract("Tighten to 25 ft-lb, check 30-35 psi and wait 10 seconds.")
    assert [(q.value, q.high, q.unit) for q in qs] == [(25, None, "foot * force_pound"), (30, 35.0, "psi"), (10, None, "second")]


def test_correct_conversion_is_supported():
    assert statuses("Tighten to 33.9 Nm.", "Tighten to 25 ft-lb.") == ["supported"]


def test_same_number_different_unit_is_flagged():
    v = check("Tighten to 25 Nm.", "Tighten to 25 ft-lb.")
    assert v[0].status == "unit_mismatch" and "different unit" in v[0].proof


def test_wrong_value_is_flagged_with_difference():
    v = check("The maximum supply is 15 v.", "maximum supply voltage : 12 v . do not exceed .")
    assert v[0].status == "mismatch" and "%" in v[0].proof


def test_value_inside_a_range_is_supported():
    assert statuses("Inflate to 32 psi.", "recommended pressure is 30 - 35 psi .") == ["supported"]


def test_bare_number_absent():
    assert statuses("Press the button 4 times.", "press and hold the button for 3 seconds .") == ["absent"]


@pytest.mark.parametrize("claim,evidence", [("Wait 2 minutes.", "wait 120 seconds ."), ("It holds 6 quarts.", "capacity 5.7 l .")])
def test_unit_conversions_supported(claim, evidence):
    assert statuses(claim, evidence) == ["supported"]


def test_temperature_offset_units():
    assert statuses("Below 32 degrees the system turns off.", "below 0 c the system turns off .") == ["supported"]


def test_canonicalize_rewrites_converted_values_only():
    from claimgate.quantities import canonicalize
    new, notes = canonicalize("Tighten the lug nuts to 176 Nm.", "tighten lug nuts to 130 ft-lb .")
    assert "130 ft-lb" in new and notes
    same, none = canonicalize("Tighten the lug nuts to 200 Nm.", "tighten lug nuts to 130 ft-lb .")
    assert same == "Tighten the lug nuts to 200 Nm." and not none


def test_contradiction_needs_an_evidence_sentence_about_the_same_thing():
    ev = "Check the lug nut torque again after the first 25 miles. The spare should not be driven above 50 mph."
    assert statuses("Rotate your tires every 5,000 miles.", ev) == ["absent"]
    assert statuses("The spare can be driven at 70 mph.", ev) == ["mismatch"]


def test_number_words_with_single_letter_are_not_units():
    assert [q.unit for q in extract("Check it once a month and two a year.")] == [None]
