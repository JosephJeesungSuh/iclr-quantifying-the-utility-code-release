# Third-party source notices

Submission-specific code uses the root BSD 3-Clause license, with the holder
shown as Anonymous authors for double-blind review.

- `sftuser_training/llama-cookbook/` contains a customized subset of Meta's
  llama-cookbook. Its supplied license and individual source notices are retained.
- `rl_training/` adapts the user-environment integration from
  [SkyRL](https://github.com/NovaSky-AI/SkyRL). Its supplied Apache-2.0 license is
  retained as `rl_training/LICENSE`. The framework itself is installed separately
  at the public commit specified in `rl_training/install.py`.
- Individual files retain third-party copyright notices and attribution links.
  File-specific notices continue to apply.

Adaptations include training configuration, preprocessing, sequence packing,
token initialization, evaluation statistics, and study data handling. Public datasets, model weights, and their licenses are not bundled or
replaced by the root source-code license.
