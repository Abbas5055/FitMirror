"""Unit tests for the sizing engine. These match the spec assertions."""

from fitmirror.sizing import recommend


def test_mens_kurta_chest_96_is_M():
    r = recommend("mens_kurta", chest_cm=96.0)
    assert r.size == "M", f"expected M, got {r.size}"


def test_womens_kurta_86_70_94_is_S():
    r = recommend("womens_kurta", chest_cm=86.0, waist_cm=70.0, hip_cm=94.0)
    assert r.size == "S", f"expected S, got {r.size}"


def test_saree_blouse_chest_86_is_34():
    r = recommend("saree_blouse", chest_cm=86.0)
    assert r.size == "34", f"expected 34, got {r.size}"


def test_anarkali_drops_hip_dim():
    r = recommend("womens_anarkali", chest_cm=86.0, waist_cm=70.0, hip_cm=130.0)
    assert "hip" not in r.per_dimension


def test_disagreeing_dims_round_up():
    # chest 96 -> L (91.1-96.0), waist 65 -> XS (62-67), hip 95 -> S (91.1-95.0).
    # Spread > 1 -> rounds up to L, low confidence.
    r = recommend("womens_kurta", chest_cm=96.0, waist_cm=65.0, hip_cm=95.0)
    assert r.confidence == "low", f"expected low, got {r.confidence}"
    assert r.size == "L", f"expected L, got {r.size}"


def test_extreme_chest_falls_to_nearest():
    # Below smallest XS -> nearest is XS
    r = recommend("mens_kurta", chest_cm=70.0)
    assert r.size == "XS"
    # Above largest XXXL -> nearest is XXXL
    r = recommend("mens_kurta", chest_cm=140.0)
    assert r.size == "XXXL"


if __name__ == "__main__":
    # Simple ad-hoc runner so the file is also runnable without pytest.
    test_mens_kurta_chest_96_is_M()
    test_womens_kurta_86_70_94_is_S()
    test_saree_blouse_chest_86_is_34()
    test_anarkali_drops_hip_dim()
    test_disagreeing_dims_round_up()
    test_extreme_chest_falls_to_nearest()
    print("All sizing tests passed.")
