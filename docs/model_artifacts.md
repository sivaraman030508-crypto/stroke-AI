# Model Artifacts Documentation

## Runtime Artifacts (project-local)

The original downloads are extracted PyTorch zip archives (directories, not files),
so loadable weights-only checkpoints were rebuilt into `artifacts/`:

- `artifacts\stroke_type_resnet18.pt` — Model A weights (OrderedDict of tensors)
- `artifacts\model_a_metadata.json` — Model A metadata (copy of original)
- `artifacts\STROKE_AI_Model_B_Final.pt` — Model B weights (unwrapped state dict)
- `artifacts\STROKE_AI_Model_B_Final_metadata.json` — Model B metadata (copy, with
  `image_size: 224` and `input_channels: 3` added)

The original downloads are left untouched (not committed).

## Model A (ResNet18 - Stroke Type Classification)

- **Task**: normal_ischemic_hemorrhagic_classification
- **Classes**: `normal`, `ischemic`, `hemorrhagic`
- **Input size**: 224
- **Input channels**: 1 (grayscale)
- **Architecture**: timm `resnet18` feature extractor (`num_classes=0`) +
  `classifier` head `Linear(512→256) → ReLU → Linear(256→3)`
- **Checkpoint**: `artifacts/stroke_type_resnet18.pt`
- **Metadata**: `artifacts/model_a_metadata.json`
- **Version**: `stroke-type-triage-v1`
- **Validation accuracy**: 0.9586 (from metadata)
- **Dataset**: ozguraslank/brain-stroke-ct-dataset

## Model B (ResNet18 - Hemorrhage Subtype Classification)

- **Task**: hemorrhage_subtype_classification
- **Subtype labels**: `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`
- **Input size**: 224 (resolved from checkpoint/architecture)
- **Input channels**: 3 (RGB-style; grayscale slices are repeated across channels)
- **Architecture**: ResNet18 with `fc: Linear(512→5)`
- **Checkpoint**: `artifacts/STROKE_AI_Model_B_Final.pt`
- **Metadata**: `artifacts/STROKE_AI_Model_B_Final_metadata.json`
- **Version**: `STROKE_AI_Model_B_Final`
- **Best validation mAP**: 0.7811 (from metadata)
- **Final training loss**: 0.3143
- **Final validation loss**: 0.171
- **Experiment details**: imbalance_weighted_finetuning

## Verification

- [x] Both checkpoints load on CPU through `create_model_a_service()` / `create_model_b_service()`
- [x] Dummy-tensor inference returns valid probability distributions (sum ≈ 1.0)
- [x] Model B input channels inferred from checkpoint (3) and grayscale inputs
      automatically repeated
- [x] Model B key prefixing handled (`conv1.*` / `fc.*` mapped onto `backbone.*`)
- [x] Metadata for both models complete in `artifacts\`

## Historical Issues (resolved)

1. **Checkpoints were extracted directories** — rebuilt into weights-only `.pt`
   files under `artifacts\`.
2. **Model B metadata missing `image_size` / `input_channels`** — resolved by the
   artifacts metadata copy (`224`, `3`) and dynamic channel inference from the
   checkpoint.
3. **Scaffolded model class did not match trained weights** — Model A now matches
   the `backbone` + `classifier` head layout; Model B matches plain ResNet18 keys.
