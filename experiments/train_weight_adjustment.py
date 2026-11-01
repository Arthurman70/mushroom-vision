"""Bounded, balanced weight fitting from frozen AI reviews; never activation.

Both resolution recipes share the same initialization, examples, paired order,
optimizer and final-epoch rule. No reserved regression examples are read here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_ai_corpus_candidate import (
    digest, ensure_background_proposals, evaluate, load_records,
    negative_regions, tensor_example,
)
from experiments.weight_recipe import (
    build_recipe_model, config_for, load_recipe_checkpoint, paired_epoch,
)
from experiments.weight_diagnostics import diagnose
from vision.training_pipeline import MODEL_NAME, WEIGHT_SHA256
from vision.video_ai_pass import atomic_json


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def examples_for(rows):
    examples, crops, rejected = [], Counter(), 0
    for row in rows:
        if row["split"] != "train":
            continue
        examples.append({"id": row["id"], "row": row, "positive": bool(row["boxes"])})
        regions, count = negative_regions(row)
        rejected += count
        for region in regions[:2]:
            examples.append({"id": region["id"], "row": row, "positive": False, "crop": region["bounds"]})
            crops[region["kind"]] += 1
    return examples, dict(crops), rejected


def background_fallback(model, counter):
    def hook(module, inputs, result):
        if not module.training:
            return result
        proposals, count = ensure_background_proposals(result[0], inputs[0].image_sizes, inputs[2])
        counter["inserted"] += count
        return proposals, result[1]
    return model.rpn.register_forward_hook(hook)


def train_step(model, optimizer, examples, root, augment):
    import torch
    items = [tensor_example(root, example, augment=augment) for example in examples]
    values = model([item[0] for item in items], [item[1] for item in items])
    loss = sum(values.values())
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss; this candidate cannot be used")
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
    if not torch.isfinite(norm):
        raise ValueError("Non-finite gradients; this candidate cannot be used")
    optimizer.step()
    return float(loss.detach()), {key: float(value.detach()) for key, value in values.items()}


def sanity_row(rows):
    """Select an easy, large training-only original without inspecting model output."""
    candidates = [row for row in rows if row["split"] == "train" and len(row["boxes"]) == 1]
    if not candidates:
        raise ValueError("A single-target training original is required for the fit diagnostic")
    def size(row):
        b = row["boxes"][0]["bboxPixelsXYXY"]
        return min((b[2]-b[0])/row["width"], (b[3]-b[1])/row["height"]), row["id"]
    return max(candidates, key=size)


def tiny_fit(rows, config, output, seed, root, steps=80):
    import torch
    seed_all(seed)
    row = sanity_row(rows)
    model = build_recipe_model(config, pretrained=True)
    fallback = Counter()
    handle = background_fallback(model, fallback)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=.001, momentum=.9, weight_decay=.0005)
    history = []
    model.train()
    model.backbone.body.eval()
    for index in range(steps):
        loss, parts = train_step(model, optimizer, [{"row": row, "positive": True}], root, False)
        if (index+1) % 20 == 0:
            history.append({"step": index+1, "loss": loss, "components": parts})
            print(f"Training-only sanity fit: {index+1}/{steps} updates, loss={loss:.4f}", flush=True)
    handle.remove()
    diagnostics = diagnose(model, [row], root)
    # The separate simple scorer uses the same operating threshold as deployment.
    metrics, _ = evaluate(model, [row], root)
    result = {"schema": "mushroom-vision.weight-sanity-fit.v1", "imageId": row["id"],
              "sourceSplit": "train", "optimizerSteps": steps, "history": history,
              "metrics": metrics, "diagnostics": diagnostics,
              "passed": metrics["overall"]["truePositives"] >= 1,
              "discardedBeforeMainTraining": True, "independentAccuracyEvidence": False}
    atomic_json(output/"sanity-fit.json", result)
    return result


def train(manifest, inventory, ledger, output, recipe="balanced-320", epochs=6,
          learning_rate=.001, seed=8251, sanity=False, root=ROOT):
    import torch
    if type(epochs) is not int or not 1 <= epochs <= 12 or not .0001 <= learning_rate <= .005:
        raise ValueError("Use 1–12 passes and a learning rate from .0001 to .005")
    config = config_for(recipe)
    rows, provenance = load_records(manifest, inventory, ledger, root)
    examples, crops, rejected = examples_for(rows)
    pairs = paired_epoch(examples, seed+1)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/"inputs"
    inputs.mkdir()
    for source, name, key in ((manifest, "training-export.json", "manifestSha256"),
                              (inventory, "inventory.json", "inventorySha256"),
                              (ledger, "ledger.json", "ledgerSha256")):
        shutil.copyfile(source, inputs/name)
        if digest(inputs/name) != provenance[key]:
            raise ValueError("Source metadata changed during freezing")
    atomic_json(output/"accepted-records.json", rows)
    protocol = {"schema": "mushroom-vision.weight-fit-protocol.v1", "frozenBeforeTraining": True,
        "createdAt": datetime.now(timezone.utc).isoformat(), **provenance,
        "model": MODEL_NAME, "modelConfig": config, "initializationSha256": WEIGHT_SHA256,
        "initialization": "Official COCO weights with a seeded new two-class head, not the failed checkpoint",
        "epochs": epochs, "plannedSteps": epochs*len(pairs), "batchSize": 2,
        "positiveExamplesPerEpoch": len(pairs), "negativeExamplesPerEpoch": len(pairs),
        "uniquePositiveExamples": sum(example["positive"] for example in examples),
        "uniqueNegativeExamples": sum(not example["positive"] for example in examples),
        "sampling": "Every negative once per pass; one source-cycled positive paired with each negative; all positives visited",
        "learningRate": learning_rate, "optimizer": "SGD momentum=.9 weight_decay=.0005 gradient_clip=10",
        "lossWeights": {"loss_classifier": 1, "loss_box_reg": 1, "loss_objectness": 1, "loss_rpn_box_reg": 1},
        "selection": "Fixed final epoch; no checkpoint or threshold selection using regression outcomes",
        "seed": seed, "cpuThreads": 4, "scoreThreshold": .3, "iouThreshold": .5,
        "trainingFrames": len(training), "trainingSources": len({r["sourceGroupId"] for r in training}),
        "trainingBoxes": sum(len(r["boxes"]) for r in training), "hardNegativeCrops": crops,
        "negativeRegionsRejectedForMushroomOverlap": rejected, "validationFrames": len(validation),
        "sanityFitSteps": 80 if sanity else 0, "sanitySelection": "Largest fractional single-target training original",
        "labelAuthority": "ai-visual-review", "humanVerified": False, "activationEligible": False,
        "testUsedForSelection": False, "diagnosticScoreCurveSelectsThreshold": False}
    atomic_json(output/"protocol.json", protocol)
    if sanity:
        sanity_result = tiny_fit(rows, config, output, seed, root)
        if not sanity_result["passed"]:
            atomic_json(output/"report.json", {"status": "sanity_failed", "protocol": protocol,
                "activationEligible": False, "productionModelChanged": False,
                "reason": "The tiny training-only fit could not match its own target; inspect proposal and score diagnostics first"})
            return {"status": "sanity_failed"}
    seed_all(seed)
    model = build_recipe_model(config, pretrained=True)
    initial_head = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    initial_head_hash = hashlib.sha256(initial_head.numpy().tobytes()).hexdigest()
    fallback = Counter()
    hook = background_fallback(model, fallback)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=learning_rate, momentum=.9, weight_decay=.0005)
    history, completed = [], 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        model.backbone.body.eval()
        losses, components = [], Counter()
        for positive, negative in paired_epoch(examples, seed+epoch):
            loss, parts = train_step(model, optimizer, [positive, negative], root, True)
            losses.append(loss)
            components.update(parts)
            completed += 1
            if completed % 20 == 0:
                progress = {"completedSteps": completed, "plannedSteps": protocol["plannedSteps"],
                            "epoch": epoch, "epochs": epochs, "recipe": recipe, "elapsedSeconds": time.perf_counter()-started}
                atomic_json(output/"progress.json", progress)
                print(f"{recipe}: {completed}/{protocol['plannedSteps']} updates, loss={loss:.4f}", flush=True)
        metrics, _ = evaluate(model, validation, root)
        history.append({"epoch": epoch, "meanLoss": sum(losses)/len(losses),
            "lossComponents": {key: value/len(losses) for key, value in components.items()},
            "validation": metrics})
        atomic_json(output/"history.json", history)
        print(f"{recipe}: pass {epoch}/{epochs}, validation matches={metrics['overall']['truePositives']}, false alerts={metrics['overall']['falsePositives']}", flush=True)
    hook.remove()
    delta = (model.roi_heads.box_predictor.cls_score.weight.detach()-initial_head).abs()
    checkpoint = {"schema": "mushroom-vision.weight-adjusted-checkpoint.v1", "architecture": MODEL_NAME,
        "state_dict": model.state_dict(), "modelConfig": config, "epoch": epochs, "scoreThreshold": .3,
        "reviewerKind": "ai", "humanVerified": False, "activationEligible": False,
        "trainingManifestSha256": provenance["manifestSha256"], "inventorySha256": provenance["inventorySha256"],
        "ledgerSha256": provenance["ledgerSha256"]}
    torch.save(checkpoint, output/"candidate.pt")
    # Reload the saved artifact, including its non-tensor configuration, for all final scoring.
    restored, _ = load_recipe_checkpoint(output/"candidate.pt")
    diagnostics = diagnose(restored, rows, root)
    atomic_json(output/"fit-diagnostics.json", diagnostics)
    metrics, predictions = evaluate(restored, validation, root)
    atomic_json(output/"validation-predictions.json", predictions)
    report = {"schema": "mushroom-vision.weight-adjusted-fit-report.v1", "status": "completed",
        "protocol": protocol, "optimizerSteps": completed, "selectedEpoch": epochs, "history": history,
        "validation": metrics, "initialHeadSha256": initial_head_hash,
        "maxClassifierWeightChange": float(delta.max()), "meanClassifierWeightChange": float(delta.mean()),
        "emptyNegativeProposalFallbacks": fallback["inserted"], "checkpointSha256": digest(output/"candidate.pt"),
        "elapsedSeconds": time.perf_counter()-started, "labelAuthority": "ai-visual-review", "humanVerified": False,
        "activationEligible": False, "productionModelChanged": False, "testEvaluationPerformed": False,
        "limitations": ["AI labels remain provisional; no human confirmation was created.",
            "Validation has one positive editorial frame and cannot establish woodland performance.",
            "Equal positive/negative image exposure is not equal foreground/background anchor counts.",
            "The fixed final checkpoint avoids cherry-picking the repeatedly inspected regression set.",
            "Larger input resolution increases computation, not original optical detail."]}
    atomic_json(output/"report.json", report)
    atomic_json(output/"progress.json", {"status": "completed", "completedSteps": completed, "plannedSteps": completed})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "inventory", "ledger", "output"):
        parser.add_argument("--"+argument, type=Path, required=True)
    parser.add_argument("--recipe", choices=("balanced-320", "balanced-640"), default="balanced-320")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--sanity-fit", action="store_true")
    args = parser.parse_args()
    result = train(args.manifest, args.inventory, args.ledger, args.output, args.recipe, args.epochs, sanity=args.sanity_fit)
    print(result["status"], flush=True)
