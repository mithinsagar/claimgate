"""Property-based tests for the quantity checker: invariants over random values, units and text."""
from hypothesis import given, settings
from hypothesis import strategies as st

from claimgate.quantities import UREG, canonicalize, check, extract

PAIRS = [("ft-lb", "foot * force_pound", "N·m", "newton * meter"), ("psi", "psi", "kPa", "kilopascal"),
         ("mph", "mile / hour", "km/h", "kilometer / hour"), ("miles", "mile", "km", "kilometer"), ("quarts", "quart", "L", "liter"),
         ("inches", "inch", "cm", "centimeter"), ("lbs", "pound", "kg", "kilogram"), ("gallons", "gallon", "L", "liter")]
values = st.floats(min_value=0.5, max_value=5000, allow_nan=False, allow_infinity=False)


def fmt(v: float) -> str:
    x = float(f"{v:.3g}") if v < 1000 else round(v)
    return f"{x:.0f}" if x >= 1000 else f"{x:g}"


@settings(max_examples=300, deadline=None)
@given(values, st.sampled_from(PAIRS))
def test_correct_conversion_is_always_supported(v, pair):
    u1, p1, u2, p2 = pair
    v1 = float(fmt(v))
    conv = UREG.Quantity(v1, p1).to(p2).magnitude
    verdicts = check(f"Set the limit to {fmt(v1)} {u1}.", f"Set the limit to {fmt(conv)} {u2}.")
    assert [x.status for x in verdicts] == ["supported"]


@settings(max_examples=300, deadline=None)
@given(values, st.sampled_from(PAIRS), st.floats(min_value=1.3, max_value=4.0))
def test_value_off_by_30_percent_or_more_is_flagged(v, pair, factor):
    u1, *_ = pair
    v1 = float(fmt(v))
    verdicts = check(f"Set the limit to {fmt(v1 * factor)} {u1}.", f"Set the limit to {fmt(v1)} {u1}.")
    assert [x.status for x in verdicts] == ["mismatch"]


@settings(max_examples=200, deadline=None)
@given(values, st.sampled_from(PAIRS))
def test_same_number_in_a_different_unit_is_flagged(v, pair):
    u1, p1, u2, p2 = pair
    v1 = float(fmt(v))
    if abs(UREG.Quantity(v1, p1).to(p2).magnitude - v1) <= 0.02 * v1:
        return  # the two units happen to agree at this value
    verdicts = check(f"Set the limit to {fmt(v1)} {u2}.", f"Set the limit to {fmt(v1)} {u1}.")
    assert [x.status for x in verdicts] == ["unit_mismatch"]


@settings(max_examples=500, deadline=None)
@given(st.text(max_size=300))
def test_extract_and_check_never_crash_on_arbitrary_text(text):
    extract(text)
    check(text, text[::-1])


@settings(max_examples=200, deadline=None)
@given(values, st.sampled_from(PAIRS))
def test_canonicalize_never_creates_a_mismatch(v, pair):
    u1, p1, u2, p2 = pair
    v1 = float(fmt(v))
    evidence = f"Set the limit to {fmt(UREG.Quantity(v1, p1).to(p2).magnitude)} {u2}."
    new, _ = canonicalize(f"Set the limit to {fmt(v1)} {u1}.", evidence)
    assert all(x.status == "supported" for x in check(new, evidence))
