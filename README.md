# FlexDraft

[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](./assets/paper.pdf)

This repository is the official implementation for **FlexDraft**.

## News

- (05/2025) Initial release of FlexDraft (official implementation).

## Overview

FlexDraft is a **lossless speculative decoding framework** that flexibly adapts to varying batch sizes through three key designs. It accelerates memory-bound LLM inference without quality degradation by combining lightweight block diffusion drafting with adaptive execution.

While parallel speculative decoding addresses the mutual waiting limitation of sequential methods, existing approaches either require costly continual pretraining with quality degradation or suffer from low acceptance rates. More importantly, this paradigm inherently suffers from uncertainty in both the **bonus token** and the **accepted length**, leading to draft-verification mismatch and causing throughput gains to collapse at large batch sizes. FlexDraft tackles these challenges as follows:

- **Attention Tuning** enables block diffusion drafting by tuning only the attention projectors of the final few layers on mask tokens, while keeping the autoregressive path frozen to exactly preserve the target distribution and produce high-quality drafts with minimal trainable parameters.
- **Bonus-guided Calibration** uses a lightweight MLP conditioned on the resolved bonus token to calibrate draft logits, mitigating the draft-verification mismatch caused by bonus token uncertainty.
- **Flex Decoding** dynamically switches between parallel draft-and-verify at small batch sizes and sequential draft-then-verify at large batch sizes, and further adjusts verification length based on draft confidence to eliminate redundant computation.
