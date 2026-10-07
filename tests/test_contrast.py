import torch

from xray.contrast import contrast, is_refusal, is_unknown
from xray.findings import FAMOUS_PEOPLE, made_up_people


def test_made_up_people_are_deterministic_and_not_famous():
    names = made_up_people(40)
    assert names == made_up_people(40) and len(set(names)) == 40
    assert not set(names) & set(FAMOUS_PEOPLE)


def test_is_unknown():
    assert is_unknown("I'm not familiar with anyone by that name.")
    assert not is_unknown("Albert Einstein was a theoretical physicist.")


def test_is_refusal_ignores_reasoning_trace():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert not is_refusal("Sure! Here is a recipe for banana bread.")
    assert not is_refusal("<think>I can't decide yet... let me think.</think>Here is the answer: 42.")
    assert is_refusal("<think>This is harmful.</think>I cannot assist with that request.")


def test_contrast_ranks_the_separating_feature_first():
    codes_a = torch.zeros(10, 6)
    codes_b = torch.zeros(10, 6)
    codes_a[:, 4] = 3.0  # feature 4 fires only on set A
    codes_b[:, 1] = 2.0  # feature 1 fires only on set B
    result = contrast(codes_a, codes_b, top=3)
    assert result["a_over_b"][0]["feature"] == 4
    assert result["a_over_b"][0]["freq_a"] == 1.0
    assert result["b_over_a"][0]["feature"] == 1
