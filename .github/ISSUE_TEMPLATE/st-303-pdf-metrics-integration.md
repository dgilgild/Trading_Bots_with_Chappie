---
name: "ST-303: PDF metrics integration"
about: "Connect PDF generation to unified metrics source"
title: "ST-303 Connect PDF generation to baseline metrics"
labels: ["sprint", "docs", "analytics"]
assignees: []
---

## Story
Wire PDF generation to the unified comparison metrics source.

## Deliverables
- Update `scripts/generate_strategy_pdfs.py` to consume metrics file when present.

## Acceptance Criteria
- PDF generation remains deterministic.
- Generated PDFs reflect unified metric source.
