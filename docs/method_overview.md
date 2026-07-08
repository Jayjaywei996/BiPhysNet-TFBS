# Method Overview

## Biological Motivation

Transcription factor binding site (TFBS) prediction is a central task in regulatory genomics. Many deep learning models mainly rely on sequence similarity, but protein-DNA binding is also influenced by local DNA shape, stacking interactions, and thermodynamic stability. BiPhysNet-TFBS is designed to model both the symbolic DNA sequence and the associated biophysical constraints.

## DNA Sequence Branch

The sequence branch converts each DNA segment into overlapping k-mer tokens. These tokens are encoded by a MetaBERTa-compatible DNA language model to obtain contextual representations. Attention pooling then aggregates position-level embeddings into a sequence-level vector.

This branch captures motif-like patterns and sequence-context dependencies that are useful for distinguishing TFBS and non-TFBS segments.

## 42D Biophysical Feature Branch

The physical branch uses a 42-dimensional feature vector that summarizes DNA sequence composition, dinucleotide patterns, shape-related properties, and thermodynamic characteristics.

The feature vector is projected into the same hidden space as the sequence representation and encoded with a lightweight Transformer-based physical encoder. This branch provides structural and energy-related evidence that can complement the sequence branch.

## MLM Objective

Masked language modeling (MLM) randomly masks part of the k-mer tokens and trains the sequence encoder to recover them from context. This objective helps the sequence encoder learn DNA grammar before supervised TFBS training.

## ITC Contrastive Objective

The image-text-style contrastive objective treats the sequence representation and the matching physical representation from the same DNA segment as a positive pair. Non-matching sequence-physical pairs within the same mini-batch are treated as negatives.

This objective aligns the two modalities in a shared representation space and encourages the physical branch to learn structure-aware information that is compatible with DNA sequence semantics.

## Adaptive Sequence-Structure Gating

During fine-tuning, BiPhysNet-TFBS combines sequence and physical representations using an adaptive gate:

```text
h_final = (1 - alpha) * h_seq + alpha * h_phys'
```

The gate is computed from sequence-physical interactions and similarity information. When the sequence and physical features are consistent, the model can rely more on the sequence representation. When they are inconsistent, the physical branch can provide an additional constraint.

## Downstream TFBS Classification

The fused representation is passed to a binary classification head. The model supports focal loss for class imbalance and includes a consistency term between the sequence and physical branch predictions.

The fine-tuning stage reports AUROC, AUPR, Accuracy, F1, MCC, Sensitivity, and Specificity.

## Why This Design Helps

A sequence-only model may assign high scores to negative samples that contain similar motifs but lack compatible structural properties. By adding a physical branch and a gating mechanism, BiPhysNet-TFBS can use sequence-structure consistency as a secondary decision signal. This design is especially useful for reducing sequence-similarity-driven false positives in internal benchmark experiments.
