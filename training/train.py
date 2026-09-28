#!/usr/bin/env python3
"""Train the temporal sign classifier and export TFLite.

    python training/train.py --epochs 60 --window 40

Pipeline: clips (.npz) -> body-normalized features + velocity -> resample to
WINDOW frames -> 1D-CNN + GRU -> softmax over vocab.ALL_CLASSES.

Held-out split is BY SIGNER when >= 2 signers exist (the honest metric), else
a random 80/20 clip split. Outputs go to data/models/sign_classifier.{keras,tflite}
plus a confusion matrix PNG and a labels.txt the Pi runtime reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from s2s import vocab  # noqa: E402
from s2s.clips import iter_clips  # noqa: E402
from s2s.features import FEAT_DIM, clip_to_features, mirror_lr, resample  # noqa: E402

OUT_DIR = ROOT / "data" / "models"


def load_dataset(window: int, augment_mirror: bool):
    X, y, signers = [], [], []
    for _, raw, _, meta in iter_clips():
        if meta.sign not in vocab.CLASS_TO_ID:
            continue
        variants = [raw]
        if augment_mirror:
            variants.append(mirror_lr(raw))
        for r in variants:
            X.append(resample(clip_to_features(r), window))
            y.append(vocab.CLASS_TO_ID[meta.sign])
            signers.append(meta.signer)
    if not X:
        sys.exit("No clips found under data/clips. Run scripts/collect.py first.")
    return np.stack(X), np.array(y), np.array(signers)


def split(X, y, signers, holdout_signer: str | None):
    uniq = sorted(set(signers))
    if holdout_signer is None and len(uniq) >= 2:
        holdout_signer = uniq[-1]
    if holdout_signer:
        te = signers == holdout_signer
        print(f"holding out signer '{holdout_signer}' ({te.sum()} clips) for validation")
        return X[~te], y[~te], X[te], y[te]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_te = max(1, len(X) // 5)
    te, tr = idx[:n_te], idx[n_te:]
    return X[tr], y[tr], X[te], y[te]


def build_model(window: int, n_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers as L

    inp = L.Input(shape=(window, FEAT_DIM))
    x = L.Conv1D(96, 5, padding="same", activation="relu")(inp)
    x = L.BatchNormalization()(x)
    x = L.Conv1D(96, 5, padding="same", activation="relu")(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Dropout(0.3)(x)
    x = L.GRU(96, return_sequences=False, unroll=True)(x)  # unroll => pure TFLite builtins (no Flex ops)
    x = L.Dropout(0.3)(x)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def jitter(X, rng, sigma=0.01):
    return X + rng.normal(0, sigma, X.shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--holdout-signer", default=None)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf

    X, y, signers = load_dataset(args.window, not args.no_mirror)
    print(f"dataset: {X.shape}, classes present: {len(set(y))}/{len(vocab.ALL_CLASSES)}")
    Xtr, ytr, Xte, yte = split(X, y, signers, args.holdout_signer)

    model = build_model(args.window, len(vocab.ALL_CLASSES))
    model.summary()
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    rng = np.random.default_rng(0)
    model.fit(jitter(Xtr, rng), ytr, validation_data=(Xte, yte),
              epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=2)

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nvalidation accuracy: {acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "sign_classifier.keras")
    (OUT_DIR / "labels.txt").write_text("\n".join(vocab.ALL_CLASSES) + "\n")
    (OUT_DIR / "model_config.json").write_text(json.dumps({"window": args.window, "feat_dim": FEAT_DIM}))

    # TFLite export (float32 first; try int8 later once accuracy is stable)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # must run on tflite-runtime
    tfl = conv.convert()
    (OUT_DIR / "sign_classifier.tflite").write_bytes(tfl)
    print(f"wrote {OUT_DIR/'sign_classifier.tflite'} ({len(tfl)/1024:.0f} KB)")

    # Confusion matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        pred = model.predict(Xte, verbose=0).argmax(1)
        labels = sorted(set(yte) | set(pred))
        cm = confusion_matrix(yte, pred, labels=labels)
        names = [vocab.ID_TO_CLASS[i] for i in labels]
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"val acc {acc:.3f}")
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
        fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
        print("wrote confusion_matrix.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
