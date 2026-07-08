# BiPhysNet-TFBS Pipeline

```mermaid
flowchart TD
    A[DNA sequence] --> B[k-mer tokenizer]
    B --> C[MetaBERTa-compatible DNA encoder]
    C --> D[Sequence embedding]

    A --> E[42D biophysical feature extraction]
    E --> F[Physical feature encoder]
    F --> G[Physical embedding]

    D --> H[Stage 1: MLM objective]
    D --> I[Stage 1: ITC contrastive alignment]
    G --> I

    D --> J[Stage 2: adaptive gating]
    G --> J
    J --> K[Fused representation]
    K --> L[Binary TFBS classifier]
    L --> M[TFBS / non-TFBS prediction]
```

## Text Summary

1. DNA sequences are tokenized into overlapping k-mers.
2. A DNA language-model encoder produces contextual sequence embeddings.
3. A 42D biophysical feature vector provides structural and thermodynamic evidence.
4. MLM and ITC objectives align sequence and physical representations during pretraining.
5. Adaptive gating fuses both branches during supervised fine-tuning.
6. The final classifier predicts whether the sequence is a TFBS.
```
DNA sequence
├── k-mer tokens -> sequence encoder -> sequence embedding
└── 42D features -> physical encoder -> physical embedding
                 -> MLM + ITC pretraining
                 -> adaptive gating
                 -> TFBS prediction
```
