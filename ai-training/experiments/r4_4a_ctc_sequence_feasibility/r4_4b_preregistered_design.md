# R4-4B Preregistered Self-Trained CTC Design

Research-only, not production, R4 TEST closed.

Use 16,259 TRAIN and 7,728 VALIDATION R4-3A-eligible words. The target is the manual canonical observed sequence: correct emits expected, substitution emits observed, deletion emits nothing; addition-containing words are excluded.

Input is the full MFA word span at 16 kHz with no side context. Extract 64-bin log-mels using 25 ms windows and 10 ms hop. Use a fresh R3-like convolutional encoder, remove temporal attention, change the second pool to frequency-only, average only frequency, and apply Dropout(0.2)+Linear(96,41) at every output step. One temporal pooling stage yields approximately 20 ms output spacing and zero theoretical CTC alignment failures. Class 40 is CTC blank. Initialize all trainable parameters randomly so altered feature and pooling semantics do not introduce an R3-weight-transfer confound.

Train exactly one seed-42 run for 36 epochs using batch 8, Adam 1e-4, no augmentation/sampler/auxiliary loss, gradient clipping 5.0, and CTCLoss(blank=40,reduction=mean,zero_infinity=true). Select lowest validation PER, then higher deletion F1, then earlier epoch. Decode greedily, then align expected versus decoded with unit-cost Levenshtein tie order MATCH > SUBSTITUTION > DELETE > INSERT.

Frozen matched control: 717 pairs, SHA-256 `D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798`.

Hard gates: PER <= .55; binary Macro-F1 >= .70; deletion recall >= .45; deletion F1 >= .40; substitution false-deletion <= .25; matched Macro-F1 >= .60; matched deletion F1 >= .55; each speaker with >=30 deletions recall >= .25; 3-relation Macro-F1 >= .40.

No training or validation model inference occurred in R4-4A.
