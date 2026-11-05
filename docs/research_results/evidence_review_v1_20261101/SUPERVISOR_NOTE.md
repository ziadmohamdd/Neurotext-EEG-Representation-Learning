# Supervisor review: EEG project

The project implemented local data preparation, fixed-pool sentence retrieval,
classical and neural models, reconstruction pretraining, an adapted frozen EEG
backbone, matched negative controls, reader-disjoint folds, training-only
preprocessing and versioned experiment provenance. Source/cache pairing,
padding/missingness handling, target preservation and saved-rank consistency
were checked. This is verified engineering and a reportable controlled negative
result; it is not a validated EEG-to-language decoder.

The tested waveform, spectral, ordered-word, relative-amplitude, GRU,
full-candidate-loss and word-pretraining configurations did not establish a
useful, consistent EEG contribution beyond the strongest matched controls.
Structural count/duration features retrieve repeated sentence classes much more
strongly; that result must remain separate from incremental EEG information.
Word-task accuracy is also separate from sentence retrieval.

Uncertainty remains about reader/stimulus generalization, source recording
preprocessing and physical boundaries, cross-release participant independence,
and which representations or supervision might preserve useful EEG information.
No confirmed current pairing/ranking bug explains the negative results. The
data do not establish that EEG contains no language information.

Further experiments would need a revised falsifiable hypothesis and a narrower
research question with explicit controls. Generalization claims also need
independent readers/stimuli and an untouched evaluation plan. Repeated tuning
on the existing development history cannot provide that confirmation. No new
experiment, evaluation or correspondence is authorized by this review.

See [the verified research review](README.md) and its scope-separated tables.
