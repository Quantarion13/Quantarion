flowchart TD
    subgraph INPUT["Input Layer"]
        A[Read Multi-Language Files]
        B[Semantic Tagging (TAG Layer)]
    end

    subgraph MEMORY["Memory Layer"]
        C{LUT Cache?}
        D[LUT Result]
        E[HGME Retrieval]
    end

    subgraph FUSION["Fusion & Stabilization"]
        F[φ⁴³ Fusion / Scaling]
    end

    subgraph VALIDATION["Validation & Constraints"]
        G[Constraint / Invariant Check (Kaprekar)]
    end

    subgraph OUTPUT["Output & Storage"]
        H[Save Results / Emit to Storage]
        I[Update Metrics & Dashboard]
    end

    A --> B --> C
    C -- Hit --> D --> H
    C -- Miss --> E --> F --> G --> H
    H --> I
