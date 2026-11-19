# Do saliency maps find the lesion? Explainable AI on Kvasir-Capsule

This project fine-tunes an ImageNet CNN to classify video capsule endoscopy frames and
then tests the explanations it produces. It asks three questions:

1. **Localisation.** Do Grad-CAM, Integrated Gradients and Saliency maps put their weight
   inside the clinician's bounding box, and by how much more than a map that knows nothing
   about the image?
2. **Faithfulness.** Do the maps depend on what the model learned? This is checked with
   the cascading model-parameter randomisation test of Adebayo et al. (2018).
3. **Calibration.** Can the classifier's confidence be trusted, before and after
   temperature scaling?

Every number is reported with a 95% confidence interval from a bootstrap over videos,
because frames from one video are strongly correlated.

> **Status:** the pipeline is complete and passes an end-to-end CPU smoke test.
> The full GPU results are not in yet. The Results section will be filled from
> `results/*.csv` after the run.

## Dataset and protocol

[Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/) (Smedsrud et al., 2021) has
47,238 labelled frames in 14 classes. This project uses the authors'
[official two-fold splits](https://github.com/simula/kvasir-capsule/tree/master/official_splits),
which leave out Polyp, Blood - hematin and Ampulla of Vater. That leaves **47,161 frames
in 11 classes from 43 videos**, all 336×336 px. The model is trained on one split and tested on the other,
in both directions, and the two results are averaged.

`scripts/prepare_data.py` checks every download against a pinned SHA-256 and writes
`results/data_audit.json`. The audit found:

| Class | Split 0 | Split 1 | Frames with a box (0 / 1) |
|---|---:|---:|---:|
| Angiectasia | 771 | 95 | 771 / 95 |
| Blood | 22 | 424 | 22 / 424 |
| Erosion | 345 | 161 | 345 / 161 |
| Erythematous | 132 | 27 | 90 / 27 |
| Foreign Bodies | 590 | 186 | 590 / 186 |
| Ileo-cecal valve | 2,795 | 1,394 | – |
| Lymphangiectasia | 224 | 368 | 224 / 368 |
| Normal | 15,853 | 18,485 | – |
| Pylorus | 938 | 591 | – |
| Reduced Mucosal View | 1,119 | 1,787 | – |
| Ulcer | 272 | 582 | 272 / 582 |
| **Total** | **23,061** | **24,100** | 2,314 / 1,843 |

Points to keep in mind when reading the results:

- **Class imbalance.** 72.8% of frames are Normal. Accuracy is therefore a weak measure.
  The model is selected on validation MCC, and macro-F1, balanced accuracy and MCC are
  reported alongside accuracy.
- **Videos shared between the splits.** The official splits are not fully separated by
  video. 7 videos contribute frames to both splits, each time with different classes in the two splits
  (5,052 frames in split 0, 2,892 in split 1). A model can therefore be tested on a patient it
  saw in training. Results are reported on the official test split (`subset=all`) and again
  on the test frames from the remaining videos only (`subset=unseen_videos`).
- **Very small classes.** Blood has 22 frames in split 0, and Erythematous has 27 in split 1,
  all from a single video that is also in split 0. Per-class scores for these are not
  reliable, and Erythematous is absent from the `unseen_videos` subset of split 1.
- **Two findings on one frame.** 16 rows of the index (8 frames) carry two findings. They are
  kept, as in the official splits, and flagged with `multi_finding=1`.
- **Boxes.** Metadata gives each box as four corner points. The axis-aligned box around
  them is used. Boxes are rectangles around the finding, not pixel masks.

## Method

**Classifier.** ResNet-50 pretrained on ImageNet (ResNet-18 and DenseNet-121 are also
supported), input 224×224, with flips, 90° rotations and light colour jitter.
AdamW (lr 1e-4, weight decay 1e-4) with a one-cycle schedule, 10 epochs, batch 64 and
mixed precision on GPU. The loss is cross-entropy with inverse-square-root class weights.
Validation is one of five video-grouped, class-stratified folds of the training split, so no
video is shared between training and validation. The checkpoint with the best validation MCC
is kept.

**Explanations** ([Captum](https://captum.ai/)), always for the true class:

| Method | Notes |
|---|---|
| Grad-CAM (Selvaraju et al., 2017) | last convolutional block, upsampled bilinearly |
| Integrated Gradients (Sundararajan et al., 2017) | black-image baseline, 32 steps, \|attribution\| summed over channels |
| Saliency (Simonyan et al., 2014) | \|gradient\| summed over channels |
| `random` | uniform noise: the chance level |
| `center` | Gaussian centred on the image: lesions are not uniformly placed, so a fixed centre prior can score above chance |

A method is only useful if it clearly beats both reference maps.

**Localisation metrics**, computed for each boxed lesion frame of the test split:

| Metric | Definition | Chance level |
|---|---|---|
| `pointing_hit` | the most important pixel lies inside the box (pointing game, Zhang et al., 2018) | box area fraction |
| `energy_in_box` | share of positive attribution mass inside the box (Wang et al., 2020) | box area fraction |
| `top_area_prec` | of the *k* most important pixels, *k* = box area, the share inside the box | box area fraction |

Results are given for all boxed frames, for correctly classified frames only, and for
frames from unseen videos, overall and by class. `degenerate_rate` is the share of maps with
no positive mass, such as an all-zero Grad-CAM.

**Sanity check.** Layers are re-initialised from the classifier head downwards
(fc, layer4, …, conv1). After each step, the maps of 100 test frames are compared with the
trained model's maps by Spearman rank correlation, on 4×4-pooled absolute maps. If the
correlation stays high while the model is being destroyed, the method shows image structure
rather than what the model learned.

**Calibration.** ECE and MCE (15 bins), NLL and Brier score, before and after temperature
scaling (Guo et al., 2017), with the temperature fitted on validation logits.

## Results

*Pending the GPU run.* The files to fill this section from are:

- `results/classification.csv`: accuracy, balanced accuracy, macro P/R/F1, MCC with video-bootstrap CIs, by run and subset
- `results/per_class_<run>.csv`, `figures/confusion_<run>.png`
- `results/calibration.csv`, `figures/reliability_<run>.png`
- `results/localisation_<run>.csv`, `results/localisation_by_class_<run>.csv`, `figures/explanations_<run>.png`
- `results/sanity_<run>.csv`, `figures/sanity_<run>.png`, `figures/sanity_examples_<run>.png`

## How to run

### Colab (recommended, needs a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palniraj/xai-capsule-endoscopy/blob/main/notebooks/colab_pipeline.ipynb)

Open `notebooks/colab_pipeline.ipynb` in Colab with a T4 GPU and run all cells. Checkpoints
go to Google Drive, so a disconnect only loses the current epoch.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt

python scripts/prepare_data.py                         # ~520 MB download, builds data/index.csv
python scripts/train.py --arch resnet50 --train-fold 0  # train on split 0, test on split 1
python scripts/train.py --arch resnet50 --train-fold 1
python scripts/evaluate.py runs/resnet50_fold0 runs/resnet50_fold1
python scripts/explain_eval.py --run runs/resnet50_fold0
python scripts/sanity_check.py --run runs/resnet50_fold0 --n-images 100
```

CPU smoke test (a few minutes, numbers are meaningless):

```bash
python scripts/train.py --arch resnet18 --train-fold 0 --no-pretrained --epochs 2 \
    --max-images 96 --out runs/smoke_resnet18_fold0
python scripts/evaluate.py runs/smoke_resnet18_fold0
python scripts/explain_eval.py --run runs/smoke_resnet18_fold0 --max-images 30 --ig-steps 8
python scripts/sanity_check.py --run runs/smoke_resnet18_fold0 --n-images 4 --ig-steps 8
```

## Repository layout

```
capxai/            config (paths, URLs, SHA-256, classes), data, models, metrics, explain
scripts/           prepare_data, train, evaluate, explain_eval, sanity_check
notebooks/         colab_pipeline.ipynb
results/           metric tables (CSV) and data_audit.json
figures/           confusion, reliability, explanation and sanity-check figures
data/, runs/       dataset and checkpoints (created by the scripts, not committed)
```

## Limitations

- One dataset and one capsule model. Nothing here says how the classifier or its
  explanations behave on other capsules or hospitals.
- Bounding boxes are coarse, so a map can be right about the lesion and still lose mass
  outside a tight box, or score well on a large box by chance. The chance level
  (`box_area_frac`) is reported next to every score for that reason.
- Maps explain the true class. For misclassified frames this is not what the model "used"
  for its prediction, which is why the correctly classified subset is reported separately.
- A single training seed per fold. The fold-to-fold difference is the only estimate of
  training variance.
- The sanity check uses 100 frames per run.

## References

- Smedsrud, P. H. et al. (2021). Kvasir-Capsule, a video capsule endoscopy dataset. *Scientific Data* 8, 142.
- Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Simonyan, K., Vedaldi, A. & Zisserman, A. (2014). Deep inside convolutional networks: Visualising image classification models and saliency maps. *ICLR Workshop*.
- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Wang, H. et al. (2020). Score-CAM: Score-weighted visual explanations for convolutional neural networks. *CVPR Workshops*.
- Zhang, J. et al. (2018). Top-down neural attention by excitation backprop. *IJCV* 126, 1084–1102.
- Kokhlikyan, N. et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. *arXiv:2009.07896*.

## Licence

The code is released under the MIT licence (see `LICENSE`). The Kvasir-Capsule data is
distributed by its authors under CC BY 4.0 and is not redistributed in this repository.
