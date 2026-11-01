"""Bounded research recipes for changing learned detector weights.

These helpers never alter the active detector. AI labels retain their provisional
authority. Paired training batches balance image exposure; they do not imply that
the number of foreground/background proposals or their losses are equal.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from pathlib import Path
import random

from vision.training_pipeline import MODEL_NAME, build_model

CHECKPOINT_SCHEMA = "mushroom-vision.weight-adjusted-checkpoint.v1"
CONFIG_SCHEMA = "mushroom-vision.weight-recipe-config.v1"
RECIPES = {"balanced-320": (320, 640), "balanced-640": (640, 1280)}


def config_for(recipe):
    """Return a fresh, JSON-serializable description of one supported ablation."""
    if not isinstance(recipe, str) or recipe not in RECIPES:
        raise ValueError("Unsupported weight recipe")
    minimum, maximum = RECIPES[recipe]
    return {"schema": CONFIG_SCHEMA, "recipe": recipe, "minSize": minimum,
            "maxSize": maximum, "backboneFrozen": True,
            "diagnosticScoreFloor": .01, "decisionScoreThreshold": .3,
            "anchorPolicy": "pretrained-default", "rpnPolicy": "pretrained-default"}


def validate_config(config):
    """Reject silent configuration drift, including unknown inference options."""
    if not isinstance(config, dict):
        raise ValueError("Model config must be an explicit recipe object")
    expected = config_for(config.get("recipe"))
    if (set(config) != set(expected) or config != expected
        or type(config["minSize"]) is not int or type(config["maxSize"]) is not int
        or config["backboneFrozen"] is not True
        or type(config["diagnosticScoreFloor"]) not in (int, float)
        or type(config["decisionScoreThreshold"]) not in (int, float)):
        raise ValueError("Model config differs from the frozen weight recipe")
    return expected


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
        or any(type(number) not in (int, float) or not math.isfinite(number) for number in value)
        or not 0 <= value[0] < value[2] or not 0 <= value[1] < value[3]):
        raise ValueError("Invalid example box")
    return value


def _validate_examples(examples):
    if not isinstance(examples, (list, tuple)) or not examples:
        raise ValueError("Paired batches require an explicit example list")
    seen, classes = set(), {True: [], False: []}
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Invalid training example")
        identity, positive, row = example.get("id"), example.get("positive"), example.get("row")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("Missing or duplicate training example identity")
        seen.add(identity)
        if type(positive) is not bool or not isinstance(row, dict):
            raise ValueError("Training examples require explicit positive/negative labels")
        if (row.get("split") != "train" or not isinstance(row.get("sourceGroupId"), str)
            or not row["sourceGroupId"].strip()):
            raise ValueError("Training examples require a training source group")
        boxes = row.get("boxes")
        if not isinstance(boxes, list):
            raise ValueError("Training examples require complete box labels")
        truths = [_box(item.get("bboxPixelsXYXY")) if isinstance(item, dict) else _box(None)
                  for item in boxes]
        crop = example.get("crop")
        if crop is not None:
            crop = _box(crop)
            if positive:
                raise ValueError("Positive crops are not part of this weight ablation")
            if any(min(crop[2], truth[2]) > max(crop[0], truth[0])
                   and min(crop[3], truth[3]) > max(crop[1], truth[1]) for truth in truths):
                raise ValueError("Negative crop overlaps a labeled mushroom")
        elif positive != bool(truths):
            raise ValueError("Training example label contradicts its complete boxes")
        classes[positive].append(example)
    if not classes[True] or not classes[False]:
        raise ValueError("Paired batches need positive and negative examples")
    if len(classes[True]) > len(classes[False]):
        raise ValueError("This recipe requires at least as many negative as positive examples")
    return classes


def _source_order(examples, rng):
    """Visit each item once, rotating sources while their queues remain nonempty."""
    groups = defaultdict(list)
    for example in sorted(examples, key=lambda item: item["id"]):
        groups[example["row"]["sourceGroupId"]].append(example)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    ordered = []
    while keys:
        for key in keys[:]:
            ordered.append(groups[key].pop())
            if not groups[key]:
                keys.remove(key)
    return ordered


def paired_epoch(examples, seed):
    """One positive and one negative per batch, with every negative visited once.

    Positive images cycle in complete, source-interleaved permutations. For the
    frozen 26-positive/76-negative corpus this returns 76 batches: all 102 unique
    examples are seen, and each positive is seen either two or three times.
    Input order does not affect the seeded order. No input labels are modified.
    """
    if type(seed) is not int:
        raise ValueError("Sampling seed must be an integer")
    classes = _validate_examples(examples)
    rng = random.Random(seed)
    negatives = _source_order(classes[False], rng)
    positives = []
    while len(positives) < len(negatives):
        positives.extend(_source_order(classes[True], rng))
    return list(zip(positives[:len(negatives)], negatives))


def build_recipe_model(config, pretrained=True):
    """Apply non-tensor inference settings as well as the existing architecture."""
    config = validate_config(config)
    if type(pretrained) is not bool:
        raise ValueError("Pretrained initialization must be explicit")
    model = build_model(pretrained=pretrained)
    model.transform.min_size = (config["minSize"],)
    model.transform.max_size = config["maxSize"]
    model.roi_heads.score_thresh = config["diagnosticScoreFloor"]
    # build_model already freezes the body; retain this invariant explicitly.
    for parameter in model.backbone.body.parameters():
        parameter.requires_grad = False
    return model


def load_recipe_checkpoint(path):
    """Restore an AI-only inactive research model, including its resize recipe.

    Returns ``(eval_model_on_cpu, metadata_without_state_dict)``. The model emits
    diagnostics down to 0.01; callers must still score decisions at 0.3.
    """
    import torch
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(data, dict) or data.get("schema") != CHECKPOINT_SCHEMA
        or data.get("architecture") != MODEL_NAME or data.get("reviewerKind") != "ai"
        or data.get("humanVerified") is not False or data.get("activationEligible") is not False
        or type(data.get("scoreThreshold")) not in (int, float) or data["scoreThreshold"] != .3
        or ("labelAuthority" in data and data["labelAuthority"] != "ai-visual-review")
        or ("productionModelChanged" in data and data["productionModelChanged"] is not False)
        or not isinstance(data.get("state_dict"), Mapping) or not data["state_dict"]):
        raise ValueError("Unsupported or incorrectly authorized research checkpoint")
    config = validate_config(data.get("modelConfig"))
    model = build_recipe_model(config, pretrained=False)
    model.load_state_dict(data["state_dict"], strict=True)
    model.eval()
    return model, {key: value for key, value in data.items() if key != "state_dict"}
