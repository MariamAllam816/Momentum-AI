# Momentum AI — Image Analysis

Upload any photo and get a multi-model breakdown: what place/scene it was taken in, who's in it and how they feel, what objects are present, and — after analyzing a few images — personalized, context-aware location suggestions based on your patterns (e.g. "you seem to enjoy dining and social settings → here are restaurant/mountain spots in Egypt you might like").

## What it does

For each uploaded image, the pipeline runs three parallel analyses and combines them into one report:

1. **Place / scene detection** — identifies the type of location (e.g. `restaurant`, `library`, `kayaking`) with a confidence score, plus a finer-grained scene label and inferred context clues (`dining`, `social`, `work`, ...)
2. **Emotion analysis** — detects faces and classifies emotion across 7 classes (angry, disgust, fear, happy, neutral, sad, surprise); gracefully skips this step with a warning when no face is present
3. **Object detection** — lists detected objects with confidence scores and renders a horizontal bar chart

After a batch of images, it aggregates the accumulated context clues and generates personalized location suggestions with a short rationale for each.

## Example output

| Image | Detected Place | Confidence | Dominant Emotion | Key Objects |
|---|---|---|---|---|
| Burger & fries plate | restaurant | 37.78% | *(no face — skipped)* | fork, bottle, sandwich, bowl, dining table |
| Group photo in library | library | 87.41% | happy (74.25%) | book, cell phone |
| Kayaking on a river | kayaking | 77.30% | neutral (56.39%) | boat, person |

Based on these, the system suggested **mountain** and **restaurant** locations in Egypt, citing the user's apparent enjoyment of dining, social, and work-related activities.

## Tech stack *(inferred from output — please confirm/correct once source is shared)*

- Python, Jupyter Notebook
- Object detection: **YOLOv8/YOLOv5** or a COCO-pretrained torchvision model (labels match COCO classes exactly)
- Emotion recognition: **DeepFace** (7-class emotion set + face-detection-gate behavior matches DeepFace's default)
- Scene classification: an ImageNet-pretrained CNN (ResNet/MobileNet) with a custom label → place-category + context-clue mapping layer
- matplotlib for the result charts

## Known limitations

- **Place-detection confidence can be low and slightly off** — the restaurant example shows only 37.78% confidence with the underlying scene label ("cheeseburger") not matching the broader place label ("restaurant"), which suggests the mapping logic from raw classifier output to place category needs tuning.
- Emotion analysis depends entirely on a face being detected in the frame; group or busy photos with small/occluded faces may under-report.
- Location suggestions are only as good as the accumulated context clues — a single ambiguous image could skew a personalized suggestion.

## Getting started

```bash
git clone https://github.com/<your-username>/momentum-ai-image-analysis.git
cd momentum-ai-image-analysis
pip install -r requirements.txt
jupyter notebook
```

## License

MIT
