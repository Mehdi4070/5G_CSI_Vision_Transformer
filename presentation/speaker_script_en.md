# Four-speaker presentation script — 5G CSI Vision Transformer

Target duration: **13 to 15 minutes**, excluding questions.

## Suggested distribution

| Speaker | Slides | Part | Approximate duration |
|---|---:|---|---:|
| **Speaker 1** | 1 to 4 | Introduction, system overview, and CSI principle | about 3 min 10 |
| **Speaker 2** | 5 to 7 | Scenario, preprocessing, and architectures | about 3 min 35 |
| **Speaker 3** | 8 to 11 | Protocol, results, multipath, and noise | about 4 min 10 |
| **Speaker 4** | 12 to 15 | System parameters, discussion, and conclusion | about 3 min 15 |

Replace “Speaker 1,” etc. with your names. Every speaker should understand the
overall conclusion and the main figures, even when they do not appear in that
speaker's assigned part.

The script is intentionally more detailed than what should be memorized word
for word. Remember the main idea of each paragraph and speak naturally.

# Speaker 1 — Introduction and principle

## Slide 1 — Title — 20 seconds

> Hello. We are going to present our project entitled “CSI-based user
> equipment localization in 5G networks.” The objective is to use Channel
> State Information as a radio fingerprint and learn the relationship between
> this fingerprint and a user's position with a Vision Transformer.

Transition:

> I will begin by presenting the problem and our experimental objectives.

## Slide 2 — Problem and objectives — 55 seconds

> For each user, we have a complex CSI matrix organized according to the base
> station antennas and the OFDM subcarriers. We aim to directly predict the
> continuous x and y coordinates. This is therefore a regression problem, not
> a classification problem.
>
> CSI is useful because it strongly depends on position. Distances, angles,
> reflections, and delays change when the user moves. Our objectives are to
> generate data with DeepMIMO, build a leakage-free preprocessing pipeline,
> compare a CNN, a ViT, and a hybrid model, and then study multipath, noise,
> and the main system parameters.

Transition:

> Before examining the CSI in detail, here is the complete pipeline that we
> implemented.

## Slide 3 — Main system steps — 55 seconds

> Our system contains seven steps. First, we generate the channels with
> DeepMIMO. We then validate the dimensions, positions, and numerical values.
> Next, the complex CSI is converted into a real-valued tensor. We perform a
> spatial train, validation, and test split, and fit the normalization using
> the training data only.
>
> The same data protocol is used to train the CNN, the ViT, and the hybrid
> architecture. Finally, we evaluate the models in meters and perform the
> additional studies on multipath, noise, antennas, and subcarriers. Keeping
> these steps separate improves reproducibility and prevents information
> leakage.

Transition:

> To understand how this pipeline can estimate a position, let us examine what
> a CSI matrix contains.

## Slide 4 — CSI content — 60 seconds

> In our dataset, one user's CSI is a 64 by 64 complex matrix. Each row
> corresponds to one base station antenna and each column to one OFDM
> subcarrier. Every coefficient has an amplitude and a phase.
>
> The 64 antennas provide spatial diversity and contain information related to
> arrival angles. The 64 subcarriers provide frequency diversity and reveal
> the effects of different propagation delays. The amplitude mainly describes
> channel attenuation, while the phase is related to distance and propagation
> paths.
>
> The complete matrix can therefore be considered a radio fingerprint of the
> user's position. It is much richer than a single RSSI measurement.

Handover:

> We now understand why CSI can act as a position fingerprint. I will hand
> over to **[name of Speaker 2]**, who will present the scenario, the
> preprocessing, and the models.

# Speaker 2 — Data, preprocessing, and models

## Slide 5 — DeepMIMO scenario and dataset — 70 seconds

> Thank you. The dataset is generated with DeepMIMO using the O1_28 scenario.
> O1 identifies the geometry of an outdoor urban environment containing
> streets and buildings, while 28 indicates the 28 GHz carrier frequency in
> the FR2 band. We use transmitter set 4 and receiver grid 0. Within this
> grid, we select an area of approximately 20 by 20 meters containing 10,000
> users, with one position every 20 centimeters and a user height of 2 meters.
>
> The base station uses an 8 by 8 array, which gives 64 antennas spaced by half
> a wavelength. The system has 64 subcarriers over a 50 MHz bandwidth and
> retains up to 10 propagation paths. After generation, we checked the shapes,
> positions, non-finite values, zero CSI matrices, and duplicate positions. All
> 10,000 examples are valid.
>
> The base station appears far away because it is located outside the selected
> user rectangle. This is normal: the figure only shows our small study area,
> not a circular coverage radius. Ray tracing still calculates the paths
> between the station and every user in the selected region.

Transition:

> Because the raw CSI is complex-valued, it must be transformed before it can
> be used by the neural networks.

## Slide 6 — Preprocessing and spatial split — 70 seconds

> We convert every CSI matrix into three real-valued channels: logarithmic
> amplitude in decibels, cosine of the phase, and sine of the phase. The
> cosine-sine pair avoids the discontinuity between minus pi and plus pi.
>
> Each channel is standardized using a mean and standard deviation calculated
> only from the training users. The target coordinates are standardized in the
> same way. Using test-set statistics would create information leakage.
>
> The split is spatial rather than random. We use 6,800 users for training,
> 1,400 for validation, 1,400 for testing, and 400 as guard regions. A random
> split could place training and test positions only 20 centimeters apart and
> would produce an overly optimistic evaluation.

Transition:

> From these real-valued tensors, we compare three model families.

## Slide 7 — Compared architectures — 75 seconds

> The first model is a compact CNN with 86,290 parameters. It applies four
> convolutional blocks and mainly learns local patterns.
>
> Our main model is a Vision Transformer. The 64 by 64 matrix is divided into
> 8 by 8 patches, producing 64 tokens. We add a regression token and then use
> four Transformer blocks with four attention heads and an embedding dimension
> of 96. Separate positional embeddings are used for the antenna and frequency
> axes because these axes have different physical meanings.
>
> Finally, the hybrid model begins with a convolutional stem and applies
> attention to the reduced feature map. Its goal is to combine local feature
> extraction with global dependencies.

Handover:

> The three architectures are now ready to be trained and compared. I will
> hand over to **[name of Speaker 3]**, who will present the protocol and the
> experimental results.

# Speaker 3 — Protocol and experimental results

## Slide 8 — Experimental protocol — 45 seconds

> Thank you. The models predict normalized coordinates and are trained with a
> mean squared error loss. The ViT uses AdamW with an initial learning rate of
> 3 times 10 to the power of minus 4, together with a scheduler and early
> stopping. The random seed is fixed to 42, and the best ViT checkpoint is
> obtained at epoch 26.
>
> For evaluation, the coordinates are converted back into meters. We calculate
> the Euclidean localization error, its mean, median, RMSE, P90, P95, and CDF.
> Every model is evaluated on the same 1,400 test users.

Transition:

> We can now examine the main comparison.

## Slide 9 — Main results — 75 seconds

> The naive baseline, which always predicts the mean training position, has an
> error greater than 10 meters. The CNN reduces this error to 1.607 meters. The
> Vision Transformer obtains the best mean error at 0.970 meters, with a median
> of 0.883 meters and a P90 of 1.794 meters. It therefore reduces the CNN's
> mean error by 39.6 percent.
>
> The CDF confirms that this advantage is not caused by only a few examples:
> the ViT curve is generally the furthest to the left. The hybrid reaches 1.818
> meters. Its convolutional downsampling may have removed fine CSI details
> before the attention stage. This does not mean that every hybrid model is
> poor, only that this specific configuration is not optimal.

Transition:

> After the main comparison, we investigated the effect of propagation
> parameters.

## Slide 10 — Multipath richness — 65 seconds

> We generated four variants retaining at most 1, 3, 5, or 10 propagation
> paths, while keeping the same users, spatial split, ViT, and random seed.
>
> One path gives the best error at 0.715 meters. Three paths give 0.749 meters,
> five paths give 1.612 meters, and ten paths give 0.970 meters. The variation
> is therefore not monotonic. A simple channel can be easier to invert, but
> additional paths can also provide useful position information.
>
> Because we used only one training seed, the cautious conclusion is that
> multipath richness strongly changes the difficulty of the task. We cannot
> claim that fewer paths are always better.

Transition:

> We then perturbed the CSI to measure the robustness of the models.

## Slide 11 — Noise robustness — 65 seconds

> Complex additive white Gaussian noise is added directly to the CSI before
> preprocessing. The models are frozen and were trained only on clean data.
> Every SNR level is evaluated using three independent noise realizations.
>
> At 20 dB, the ViT remains close to its clean performance, with 0.997 meters.
> At 10 dB, its error increases to 1.358 meters. Even at 0 dB, when the noise
> and signal powers are equal, it remains the best model with 3.767 meters,
> compared with 9.341 meters for the CNN.
>
> Global aggregation may help the ViT use the less corrupted regions of the
> matrix. This is a plausible interpretation, not a demonstrated causal
> explanation.

Handover:

> The ViT therefore preserves its advantage when the CSI is noisy. To complete
> the analysis, I will hand over to **[name of Speaker 4]**, who will present
> the influence of system dimensions, the limitations, and the conclusion.

# Speaker 4 — Final analysis and conclusion

## Slide 12 — Antennas and subcarriers — 75 seconds

> Thank you. For the antenna study, we compare centered 4 by 4, 4 by 8, and 8
> by 8 subarrays while keeping 64 subcarriers. The error decreases from 2.305
> meters with 16 antennas to 1.306 meters with 32 antennas and 0.970 meters
> with 64 antennas.
>
> For the frequency study, we keep the 64 antennas and uniformly select 16,
> 32, or 64 subcarriers over the same 50 MHz bandwidth. The corresponding
> errors are 1.512, 1.401, and 0.970 meters.
>
> Both dimensions are useful. However, for the same number of CSI
> coefficients, keeping the antennas and reducing the frequency samples is
> less damaging. In this scenario, spatial diversity therefore appears to be
> particularly important.

Transition:

> We must now clarify what conclusions can actually be drawn from these
> results.

## Slide 13 — Discussion and limitations — 65 seconds

> The first methodological lesson is the importance of the spatial split. The
> second is that the ViT effectively uses the global structure of the CSI and
> preserves its advantage under noise. Antennas and subcarriers also provide
> complementary information.
>
> However, our study uses only one simulated scenario, one base station, a
> fixed user height, and one training seed. AWGN does not represent every
> hardware impairment, and we have not yet evaluated real CSI measurements.
>
> The correct conclusion is therefore that the ViT is the best model under our
> controlled protocol. We do not claim that it is universally superior in
> every environment.

Transition:

> I will finish with the main conclusions and future work.

## Slide 14 — Conclusion — 55 seconds

> We built a complete pipeline from ray tracing to 2D position estimation,
> using 10,000 users and a strict spatial split. The ViT reaches a mean error
> of 0.970 meters and a median error of 0.883 meters. It outperforms the CNN,
> the hybrid model, and the naive baseline, and it is also the most robust to
> noise.
>
> Future work could repeat training with multiple seeds, add noise during
> training, test generalization between scenarios, and most importantly use
> real CSI measurements and moving users.
>
> The main message is that CSI can be used as a true position sensor integrated
> into the 5G network.

Transition:

> Thank you for your attention. We are ready to answer your questions.

## Slide 15 — Questions

All four speakers should look at the jury and wait for a question. The person
who is most comfortable with the topic should answer first, and another speaker
may add information without repeating the same answer.

Keep the question-and-answer notes nearby and remember at least the following
figures: **10,000 users, a 6,800/1,400/1,400 split, 64 by 64 CSI, a ViT mean
error of 0.970 m, a CNN mean error of 1.607 m, a ViT median of 0.883 m, and a
ViT P90 of 1.794 m**.

## Presentation advice

- Do not read every value in the tables. State the trend and the best result.
- Always say “in our scenario” when discussing multipath and antenna results.
- Do not say that a 0.970 m mean error means that every user is localized
  within one meter. The P90 is 1.794 m.
- If a statement was not experimentally verified, clearly present it as a
  hypothesis or future direction.
- During questions, begin with the direct answer and then provide the
  explanation.
