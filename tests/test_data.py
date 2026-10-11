from malaria_cnn.data import assign_patient_splits, parse_mapping_file


def test_mapping_parser_handles_padded_patient_rows(tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("P001,['P001_cell_1.png', 'P001_cell_2.png'],,,\n,,,,\n", encoding="utf-8")
    rows = parse_mapping_file(mapping, "Parasitized")
    assert len(rows) == 2
    assert {row["patient_id"] for row in rows} == {"P001"}
    assert {row["label"] for row in rows} == {1}


def test_patient_group_splits_are_deterministic_and_leak_free():
    rows = []
    for group, has_parasite in [(f"healthy-{i}", False) for i in range(10)] + [(f"infected-{i}", True) for i in range(12)]:
        rows.append({"patient_id": group, "filename": f"{group}-negative.png", "label": 0})
        if has_parasite:
            rows.append({"patient_id": group, "filename": f"{group}-positive.png", "label": 1})
    first = assign_patient_splits(rows, seed=17)
    second = assign_patient_splits(rows, seed=17)
    assert [(r["patient_id"], r["split"]) for r in first] == [(r["patient_id"], r["split"]) for r in second]
    by_patient = {}
    for row in first:
        by_patient.setdefault(row["patient_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_patient.values())
    assert {row["split"] for row in first} == {"train", "validation", "test"}


def test_patient_group_split_rejects_bad_ratios():
    rows = [{"patient_id": f"h{i}", "label": 0} for i in range(4)] + [
        {"patient_id": f"p{i}", "label": 1} for i in range(4)
    ]
    try:
        assign_patient_splits(rows, ratios=(0.7, 0.2, 0.2))
    except ValueError as exc:
        assert "summing to 1" in str(exc)
    else:
        raise AssertionError("Expected invalid ratios to be rejected")
