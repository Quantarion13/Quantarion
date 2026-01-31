#diagrams:
    make -f Mermaid-Flow.mk all

#seed:
    python3 hgme_seed.py

#lut:
    python3 lut_manager.py

#metrics:
    python3 live_flow_cpu.py

#clean:
    rm -rf diagrams/*.svg metrics.json
