"""
triple2vec: Skip-gram training for SNOMED CT concept relationships.

Given a context_dict {center_AUI: [context_AUI, ...]}, trains a skip-gram
model with negative sampling to learn concept embeddings. The trained model
can then score and rank candidate missing context concepts for any center.

Usage:
    # Train
    python train_triple2vec.py --data context_dict.json --output model.pt

    # Train + predict missing contexts after training
    python train_triple2vec.py --data context_dict.json --output model.pt --predict

    # Resume from checkpoint
    python train_triple2vec.py --data context_dict.json --output model.pt \
        --resume checkpoint.pt
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Bidirectional mapping between AUI strings and integer indices."""

    def __init__(self, context_dict: dict):
        auids = set(context_dict.keys())
        for neighbors in context_dict.values():
            auids.update(neighbors)

        self.aui2idx: dict[str, int] = {aui: i for i, aui in enumerate(sorted(auids))}
        self.idx2aui: list[str] = [aui for aui, _ in sorted(self.aui2idx.items(), key=lambda x: x[1])]
        self.size = len(self.aui2idx)

    def __len__(self):
        return self.size

    def encode(self, aui: str) -> int:
        return self.aui2idx[aui]

    def decode(self, idx: int) -> str:
        return self.idx2aui[idx]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkipGramDataset(Dataset):
    """
    Produces (center_idx, positive_context_idx) pairs from context_dict.
    Negative samples are drawn on-the-fly in the model's loss to keep
    memory usage O(pairs) rather than O(pairs * neg_samples).
    """

    def __init__(self, context_dict: dict, vocab: Vocabulary):
        self.pairs: list[tuple[int, int]] = []

        for center_aui, context_auis in context_dict.items():
            if center_aui not in vocab.aui2idx:
                continue
            c_idx = vocab.encode(center_aui)
            for ctx_aui in context_auis:
                if ctx_aui in vocab.aui2idx:
                    self.pairs.append((c_idx, vocab.encode(ctx_aui)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    """
    Standard skip-gram with two separate embedding tables (center / context),
    trained with noise-contrastive negative sampling loss.

    After training, concept similarity is computed via the center embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, neg_samples: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.neg_samples = neg_samples

        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)

        # Standard word2vec initialisation
        nn.init.uniform_(self.center_emb.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.context_emb.weight)

    def forward(self, center: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        center:   (B,)   center concept indices
        positive: (B,)   known context concept indices

        Returns scalar mean loss over the batch.
        """
        batch_size = center.size(0)

        c_emb = self.center_emb(center)                              # (B, D)
        pos_emb = self.context_emb(positive)                         # (B, D)

        # Positive term: -log σ(c · pos)
        pos_score = (c_emb * pos_emb).sum(dim=1)                    # (B,)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)       # (B,)

        # Negative sampling: draw K random indices per example
        neg_idx = torch.randint(
            0, self.vocab_size, (batch_size, self.neg_samples),
            device=center.device
        )                                                            # (B, K)
        neg_emb = self.context_emb(neg_idx)                         # (B, K, D)

        # Negative term: -sum_k log σ(-c · neg_k)
        neg_scores = torch.bmm(neg_emb, c_emb.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1)  # (B,)

        return (pos_loss + neg_loss).mean()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_all_contexts(self, center_idx: int) -> torch.Tensor:
        """
        Return sigmoid scores for center_idx vs. every concept in the
        vocabulary using the context embedding matrix.

        Returns: (V,) tensor of scores in [0, 1].
        """
        c_emb = self.center_emb.weight[center_idx]          # (D,)
        scores = self.context_emb.weight @ c_emb            # (V,)
        return torch.sigmoid(scores)

    @torch.no_grad()
    def predict_missing(
        self,
        center_idx: int,
        known_context_indices: list[int],
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Rank all concepts not already in known_context_indices by their
        likelihood of being a missing context for center_idx.

        Returns list of (concept_idx, score) sorted descending.
        """
        scores = self.score_all_contexts(center_idx).cpu()

        # Suppress the center itself and already-known contexts
        suppress = set(known_context_indices) | {center_idx}
        for idx in suppress:
            scores[idx] = -1.0

        top_values, top_indices = torch.topk(scores, min(top_k, self.vocab_size))
        return list(zip(top_indices.tolist(), top_values.tolist()))

    @torch.no_grad()
    def get_center_embeddings(self) -> np.ndarray:
        return self.center_emb.weight.cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: SkipGram,
    dataset: SkipGramDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_min: float,
    num_workers: int,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_every: int,
    start_epoch: int = 0,
) -> SkipGram:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min
    )

    # Restore scheduler state if resuming mid-training
    for _ in range(start_epoch):
        scheduler.step()

    model.to(device)
    model.train()

    total_pairs = len(dataset)
    print(f"Training on {total_pairs:,} (center, context) pairs | "
          f"vocab {model.vocab_size:,} | embed {model.center_emb.embedding_dim}d | "
          f"neg_samples {model.neg_samples} | device {device}")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for center, positive in loader:
            center = center.to(device)
            positive = positive.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(center, positive)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * center.size(0)

        scheduler.step()
        avg_loss = epoch_loss / total_pairs
        elapsed = time.time() - t0

        print(f"Epoch {epoch + 1:>4}/{epochs}  loss={avg_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)

    return model


def _save_checkpoint(model, optimizer, scheduler, epoch, path: Path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Prediction report
# ---------------------------------------------------------------------------

def run_prediction(
    model: SkipGram,
    vocab: Vocabulary,
    context_dict: dict,
    top_k: int,
    max_centers: int,
):
    """Print a ranked list of predicted-missing concepts for each center."""
    model.eval()
    model.cpu()

    centers = list(context_dict.keys())[:max_centers]

    print(f"\n{'='*70}")
    print(f"Predicted missing context concepts (top {top_k} per center)")
    print(f"{'='*70}")

    for center_aui in centers:
        if center_aui not in vocab.aui2idx:
            continue
        center_idx = vocab.encode(center_aui)
        known_ctx = [vocab.encode(a) for a in context_dict[center_aui] if a in vocab.aui2idx]
        predictions = model.predict_missing(center_idx, known_ctx, top_k=top_k)

        print(f"\nCenter: {center_aui}  (known contexts: {len(known_ctx)})")
        for rank, (pred_idx, score) in enumerate(predictions, 1):
            print(f"  {rank:>3}. {vocab.decode(pred_idx)}  score={score:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train triple2vec skip-gram on SNOMED CT context dict")

    # Data
    p.add_argument("--data", required=True,
                   help="Path to context_dict JSON file")

    # Model hyper-parameters
    p.add_argument("--embed-dim", type=int, default=128,
                   help="Embedding dimensionality (default: 128)")
    p.add_argument("--neg-samples", type=int, default=10,
                   help="Negative samples per positive pair (default: 10)")

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Batch size (default: 1024)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak learning rate (default: 1e-3)")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="Minimum LR for cosine schedule (default: 1e-5)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4)")
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing / output
    p.add_argument("--output", default="triple2vec_model.pt",
                   help="Path to save the final trained model (default: triple2vec_model.pt)")
    p.add_argument("--checkpoint", default="triple2vec_checkpoint.pt",
                   help="Path for periodic checkpoints (default: triple2vec_checkpoint.pt)")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save a checkpoint every N epochs (default: 10)")
    p.add_argument("--resume", default=None,
                   help="Resume training from a checkpoint file")
    p.add_argument("--save-embeddings", default=None,
                   help="If set, also save center embeddings as a .npy file")

    # Prediction
    p.add_argument("--predict", action="store_true",
                   help="After training, print predicted missing contexts")
    p.add_argument("--predict-top-k", type=int, default=20,
                   help="Number of missing concepts to predict per center (default: 20)")
    p.add_argument("--predict-max-centers", type=int, default=10,
                   help="Max number of centers to show predictions for (default: 10)")
    p.add_argument("--predict-only", default=None,
                   help="Skip training; load model from this path and only run prediction")

    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    print(f"Loading context dict from {data_path} …")
    with data_path.open() as f:
        context_dict: dict = json.load(f)
    print(f"  {len(context_dict):,} center concepts loaded")

    vocab = Vocabulary(context_dict)
    print(f"  Vocabulary size: {vocab.size:,} unique AUIs")

    # ------------------------------------------------------------------
    # Predict-only mode
    # ------------------------------------------------------------------
    if args.predict_only:
        print(f"Loading model from {args.predict_only} for prediction …")
        checkpoint = torch.load(args.predict_only, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        # Infer shape from saved weights
        v_size, e_dim = state["center_emb.weight"].shape
        neg_s = 10  # not stored; value doesn't matter for inference
        model = SkipGram(v_size, e_dim, neg_s)
        model.load_state_dict(state)
        run_prediction(model, vocab, context_dict, args.predict_top_k, args.predict_max_centers)
        return

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    dataset = SkipGramDataset(context_dict, vocab)
    print(f"  Total (center, context) training pairs: {len(dataset):,}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = SkipGram(vocab.size, args.embed_dim, args.neg_samples)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"]
        print(f"  Resuming from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = train(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        num_workers=args.num_workers,
        device=device,
        checkpoint_path=Path(args.checkpoint),
        checkpoint_every=args.checkpoint_every,
        start_epoch=start_epoch,
    )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": {"aui2idx": vocab.aui2idx, "idx2aui": vocab.idx2aui},
            "config": {
                "vocab_size": vocab.size,
                "embed_dim": args.embed_dim,
                "neg_samples": args.neg_samples,
            },
        },
        output_path,
    )
    print(f"\nFinal model saved → {output_path}")

    if args.save_embeddings:
        emb = model.get_center_embeddings()
        np.save(args.save_embeddings, emb)
        print(f"Center embeddings saved → {args.save_embeddings}  shape={emb.shape}")

    # ------------------------------------------------------------------
    # Predict missing contexts
    # ------------------------------------------------------------------
    if args.predict:
        run_prediction(model, vocab, context_dict, args.predict_top_k, args.predict_max_centers)


if __name__ == "__main__":
    main()
