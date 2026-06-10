# Provider preprocessing notes worth using

Use the provider methods as reference baselines, not as the only preprocessing.

| Dataset | Provider method worth preserving | How to use in your project |
| :--- | :--- | :--- |
| **KARA ONE** | EEGLAB preprocessing, ocular artifact removal using blind source separation, 1–50 Hz filtering, channel mean subtraction, small Laplacian, windowing with 50% overlap, statistical/spectral features. | Implement a “provider-matched KARA baseline” and compare it against your unified deep-learning pipeline. |
| **ZuCo** | Raw and Automagic-preprocessed data are provided; preprocessing used bad-channel detection, 0.5 Hz high-pass, 49–51 Hz notch, EOG regression, MARA ICA rejection, and spherical spline interpolation. | Use provider-preprocessed ZuCo for quick baselines, but also create a unified version at 250 Hz for cross-dataset representation learning. |
| **ZuCo eye-tracking alignment** | Eye-tracking is synchronized with EEG and word fixation boundaries are available. | Use ZuCo for EEG-language pretraining, word/sentence embedding alignment, and retrieval tasks, not as imagined speech ground truth. |
