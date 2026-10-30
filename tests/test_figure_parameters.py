import pytest

from app.core.preprocess.figure_parameters import extract_parameters, parameter_tokens


def test_signed_physical_symbols_units_and_source_are_preserved():
    text = "已知 m = −2.50 kg，Fₓ=+3.20 N；θ=30°；I₀=1.2×10⁻³ A；R=5 Ω。"
    results = extract_parameters(text, "reviewed_image")
    assert [(item["symbol"], item["value"], item["unit"]) for item in results] == [
        ("m", "−2.50", "kg"), ("Fₓ", "+3.20", "N"), ("θ", "30", "°"),
        ("I₀", "1.2×10⁻³", "A"), ("R", "5", "Ω")]
    for item in results:
        assert item["source"] == "reviewed_image"
        assert text[item["start"]:item["end"]] == item["raw"]


@pytest.mark.parametrize("value", ["−1.60e-19", "＋2.0E+3", "-.5", "±0.25",
                                    "3.2×10^{-3}", "6.02 × 10^23", "4×10⁻²", "－0.5"])
def test_scientific_notation_and_signs_are_never_normalized(value):
    text = "q=" + value + " C"
    items = extract_parameters(text)
    assert len(items) == 1
    assert items[0]["symbol"] == "q"
    assert items[0]["value"] == value
    assert items[0]["unit"] == "C"
    assert items[0]["raw"] == text


def test_compound_units_and_greek_subscripts_keep_exact_spelling():
    text = "v_0=−3.5 m/s；a=9.81 m/s²；B₂=0.20 mT；C=2.2 μF；τ=5 N·m"
    items = extract_parameters(text, "annotation")
    assert [(item["symbol"], item["unit"]) for item in items] == [
        ("v_0", "m/s"), ("a", "m/s²"), ("B₂", "mT"), ("C", "μF"), ("τ", "N·m")]
    assert parameter_tokens(items) == ["速度", "加速度", "磁感应强度", "电容", "力矩"]


def test_unbound_quantities_have_no_invented_variable():
    text = "电流为12mA，电阻为3kΩ，夹角60°，标注长度2.5cm"
    items = extract_parameters(text)
    assert [(item["symbol"], item["value"], item["unit"]) for item in items] == [
        ("", "12", "mA"), ("", "3", "kΩ"), ("", "60", "°"), ("", "2.5", "cm")]


def test_unitless_assignments_do_not_infer_units_or_knowledge():
    items = extract_parameters("m=2；F=12；θ=30；I=0.5")
    assert len(items) == 4
    assert all(item["unit"] == "" for item in items)
    assert parameter_tokens(items) == []


@pytest.mark.parametrize("text", ["F=1O N", "I=l.5 A", "R=2..5 Ω", "m=?kg", "θ=O°",
                                   "F=2×10 N", "I=2e -3 A", "R=1,5 Ω", "m=2. kg"])
def test_bad_ocr_and_broken_numbers_are_not_repaired(text):
    assert extract_parameters(text) == []


@pytest.mark.parametrize("text", [None, "", "   ", "图中 F、m、θ、I 和箭头", "Ω kg mA 速度v", "公式 $F=ma$", "题号 12，第3章"])
def test_empty_or_parameter_free_text_does_not_generate_values(text):
    assert extract_parameters(text) == []


def test_units_are_not_joined_across_ocr_lines():
    items = extract_parameters("m=2\nkg\nF=−3\nN")
    assert [(item["symbol"], item["unit"]) for item in items] == [("m", ""), ("F", "")]


def test_ocr_and_user_annotation_remain_separate_even_when_values_disagree():
    ocr = extract_parameters("R=5Ω", "ocr")
    corrected = extract_parameters("R=8Ω", "annotation")
    assert ocr[0]["value"] == "5" and ocr[0]["source"] == "ocr"
    assert corrected[0]["value"] == "8" and corrected[0]["source"] == "annotation"


def test_known_unit_suffix_is_not_cut_out_of_unknown_word():
    assert extract_parameters("F=2Newton；I=3mAX；R=4Ωabc") == []


def test_repeated_observations_are_preserved_and_numeric_values_not_tokens():
    items = extract_parameters("I=2mA，I=3mA")
    assert len(items) == 2
    assert parameter_tokens(items) == ["电流"]
    assert "2" not in parameter_tokens(items)


def test_explicit_unit_semantics_do_not_guess_ambiguous_energy_or_torque():
    items = extract_parameters("5N·m")
    assert parameter_tokens(items) == []


def test_spaced_compound_units_are_literal_and_do_not_swallow_next_assignment():
    text = "ρ=1.25 kg m⁻³；τ=2.5 N m；F=3 N m=2 kg"
    items = extract_parameters(text)
    assert [(item["symbol"], item["unit"]) for item in items] == [
        ("ρ", "kg m⁻³"), ("τ", "N m"), ("F", "N"), ("m", "kg")]
    assert all(text[item["start"]:item["end"]] == item["raw"] for item in items)
