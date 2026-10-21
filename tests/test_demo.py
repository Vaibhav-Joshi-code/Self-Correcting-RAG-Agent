import pytest
from scripts.demo_scenarios import run_scenario


@pytest.mark.parametrize("name,accepted,attempts", [("success", True, 1), ("recovery", True, 2), ("failure", False, 3)])
def test_scripted_examples_exercise_the_real_graph(name, accepted, attempts):
    item = run_scenario(name)
    assert item["mode"] == "scripted_offline_demo"
    assert item["result"]["accepted"] is accepted
    assert item["result"]["attempts"] == attempts
