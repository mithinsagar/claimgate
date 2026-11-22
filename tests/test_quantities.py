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
    v = check("The maximum supply voltage is 15 V.", "Maximum supply voltage: 12 V. Do not exceed.")
    assert v[0].status == "mismatch" and "%" in v[0].proof


def test_value_inside_a_range_is_supported():
    assert statuses("Inflate to 32 psi.", "recommended pressure is 30 - 35 psi .") == ["supported"]


def test_bare_number_absent():
    assert statuses("Press the button 4 times.", "press and hold the button for 3 seconds .") == ["absent"]


@pytest.mark.parametrize("claim,evidence", [("Wait 2 minutes.", "wait 120 seconds ."), ("It holds 6 quarts.", "capacity 5.7 l .")])
def test_unit_conversions_supported(claim, evidence):
    assert statuses(claim, evidence) == ["supported"]


def test_temperature_offset_units():
    assert statuses("Below 32 degrees the system turns off.", "Below 0 C the system turns off.") == ["supported"]


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


def test_times_and_articles_are_not_quantities():
    assert extract("Assemble at 7:00 a.m. and leave at 9:30.") == []
    assert [(q.text, q.unit) for q in extract("It erupted twice in one day.")] == [("twice", None), ("one day", "day")]


def test_numbers_about_different_things_do_not_contradict():
    ev = "He served seven years in prison. They have two daughters."
    assert statuses("They have been married for 10 years.", ev) == ["absent"]


def test_claim_range_covers_evidence_value():
    assert statuses("A 20-30 foot boat fragment was found.", "A boat fragment about 30 feet long was found.") == ["supported"]


def test_ages_are_not_compared_with_durations_and_hyphenated_units_parse():
    assert statuses("Mack, a 19 year old student, was sentenced.", "Mack was sentenced to 10 years in prison.") == ["absent"]
    assert [(q.value, q.unit) for q in extract("a two-week extension and a 14-day trial")] == [(14.0, "day"), (2.0, "week")]


def test_attached_units_and_celsius_degrees():
    assert [(q.value, q.unit) for q in extract("Preheat to 250F, use a 10mm socket and a 12V supply.")] == [
        (250.0, "degF"), (10.0, "millimeter"), (12.0, "volt")]
    assert statuses("Above 35 degrees Celsius.", "High temperatures above 35 C (95 F).") == ["supported"]


def test_fraction_ranges_compound_units_and_or_clauses():
    assert [(q.value, q.high, q.unit) for q in extract("Cut 1/4 to 1/2 inch from the top.")] == [(0.25, 0.5, "inch")]
    assert [q.unit for q in extract("Density: 1738 kg/m³.")] == [None]
    ev = "Low setting would probably be 6-8 hours."
    assert statuses("Cook on high for 4 hours or on low for 6-8 hours.", ev) == ["absent", "supported"]


def test_attached_lower_case_units_are_units_but_spaced_ones_are_words():
    assert [(q.value, q.unit) for q in extract("below 10f (12c) use a 12v supply")] == [(10.0, "degF"), (12.0, "degC"), (12.0, "volt")]
    assert [q.unit for q in extract("leave at 7 a.m. today")] == [None]


def test_canonicalize_leaves_values_already_stated_in_the_same_unit():
    from claimgate.quantities import canonicalize
    new, _ = canonicalize("Service within 805 km (805 km).", "Service within 500 miles (805 km).")
    assert new == "Service within 805 km (805 km)."
