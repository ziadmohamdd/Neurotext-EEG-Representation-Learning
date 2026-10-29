# Presentation draft

Open [NeuroText_draft.pptx](NeuroText_draft.pptx) in PowerPoint. The 14-slide
deck uses the archived experimental figures and includes speaker notes with
sources and interpretation limits. It is a draft for review, not a completed
cross-dataset study. Transfer remains pending participant identity evidence.

For quick review without PowerPoint, open [the PDF](NeuroText_draft.pdf).
The PDF contains the slides; source details and speaking guidance are in the
PowerPoint notes and the report. It was rendered from the generated deck with
LibreOffice; it is a separate preview artifact, not rebuilt by the Python script.

Review the author/course information, slide readability and notes before
presenting. The 79 exported evaluations include duplicates and corruption
conditions; they are not 79 independent methods. Do not claim competitive
published-paper accuracy or successful general semantic decoding.

The transfer preparation slide uses the user's console summary: 9,233 signal
references, 594 sentence targets, a rebuilt shared cache and the expected
`blocked_pending_participant_identity` status. The full local plan JSON has not
been supplied to the repository.

To rebuild from the repository root (optional; no EEG processing):

```powershell
python -m pip install "python-pptx>=1.0,<2"
python scripts/build_presentation.py
```

This development dependency is optional and is not added to the model training
requirements. Figure regeneration is a separate step documented in
[the report](../research_results/REPORT.md). The builder replaces only the
generated deck; keep manual edits in a separately named copy before rebuilding.
