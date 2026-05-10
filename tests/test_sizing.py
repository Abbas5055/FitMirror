"""Unit tests for the sizing engine.

The chart is in inches (converted to cm internally). Borderline rule:
within 5 cm of an adjacent boundary, the size is reported as a pair.
"""

from fitmirror.sizing import recommend


def test_mens_kurta_chest_38_inch_is_M_or_S():
    # 96 cm body chest is right at the M threshold (96.52 cm). Within 5 cm.
    r = recommend("mens_kurta", chest_cm=96.0)
    assert "S" in r.size_label and "M" in r.size_label, r.size_label


def test_mens_kurta_chest_well_above_M():
    # 100 cm chest -> well into M (96.52..101.6) but within 5 cm of L -> "M or L"
    r = recommend("mens_kurta", chest_cm=100.0)
    assert "M" in r.size_label or "L" in r.size_label, r.size_label


def test_mens_kurta_clear_S():
    # 89 cm chest -> 35", clearly inside S (91.44 is upper). Borderline below S
    # so primary is XS, secondary is S.
    r = recommend("mens_kurta", chest_cm=89.0)
    # Should mention either XS or S
    assert "XS" in r.size_label or "S" in r.size_label


def test_below_smallest_returns_smallest():
    r = recommend("mens_kurta", chest_cm=70.0)
    assert "XS" in r.size_label


def test_above_largest_returns_largest():
    r = recommend("mens_kurta", chest_cm=140.0)
    assert "XXXL" in r.size_label


def test_anarkali_drops_hip():
    r = recommend("womens_anarkali", chest_cm=86.0, waist_cm=70.0, hip_cm=130.0)
    assert "hip" not in r.per_dimension


def test_high_confidence_when_all_dims_agree():
    # Construct a chest/waist/hip combination that lands cleanly inside one bucket
    # for all three dims (men's kurta L: chest 101.6+, waist 86.36+, hip 106.68+).
    r = recommend("mens_kurta", chest_cm=104.0, waist_cm=88.0, hip_cm=108.0)
    # All three should agree on "L"
    assert r.size_label == "L"
    assert r.confidence == "high"


if __name__ == "__main__":
    test_mens_kurta_chest_38_inch_is_M_or_S()
    test_mens_kurta_chest_well_above_M()
    test_mens_kurta_clear_S()
    test_below_smallest_returns_smallest()
    test_above_largest_returns_largest()
    test_anarkali_drops_hip()
    test_high_confidence_when_all_dims_agree()
    print("All sizing tests passed.")
