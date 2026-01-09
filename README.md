# WESR: Word-level Event-Speech Recognition

A comprehensive benchmark and baseline for detecting and localizing non-verbal vocal events in speech.

## Key Contributions

**WESR-Bench**: 900+ expert-annotated utterances with a novel position-aware evaluation protocol that separates ASR errors from event detection, enabling measurement of both discrete (standalone) and continuous (mixed with speech) events. [Dataset on HuggingFace](https://huggingface.co/datasets/yfish/WESR-Bench)

**Refined Taxonomy**: 21 carefully categorized vocal events with a new categorization into discrete vs. continuous events.

**Scaling**: Trained on 1,700+ hours of curated data, outperforming open-source audio-language models and commercial APIs.


## Data Format Requirements

**JSONL format:**
```json
{
  "audio": {"path": "audio_filename.wav"},
  "sentence": "predicted transcription with tags"
}
```

**JSON format:**
```json
{
  "samples": [
    {
      "audio_path": "audio_filename.wav",
      "prediction": "predicted transcription with tags"
    }
  ]
}
```

### Supported Tags

**Discrete events (15):**
- `inhale`, `cough`, `laughs`, `laughing`, `crowd_laughter`, `chuckle`, `shout`, `sobbing`, `cry`, `giggle`,`exhale`, `sigh`, `clear_throat`, `roar`, `scream`, `breathing`

**Continuous events (6):**
- `crying`, `laughing`, `panting`, `shouting`, `singing`, `whispering`

## Running

### 1. Download WESR-Bench

```bash
python eval/download.py
```

This will:
- Download the WESR-Bench dataset from Hugging Face (`yfish/WESR-Bench`)
- Save audio files to `eval/audio/`
- Create `eval/wesr_bench.jsonl` with ground truth annotations

### 2. Run Evaluation

```python
from eval.eval import evaluate, EvaluationConfig, format_output_by_tag_type

# Configure evaluation
pred_path = "your_predictions.jsonl"
true_path = ["eval/wesr_bench.jsonl"]

# Basic evaluation
config = EvaluationConfig(include_tags=None, big=False)
results = evaluate(true_path, pred_path, config, eval_type="sequence", tag_type="by_type")
print(format_output_by_tag_type(results))
```

### Command Line Usage

```bash
cd eval
python eval.py
```

Edit the `__main__` section in `eval.py` to specify your prediction file path.

### Evaluation Options

**Evaluation types:**
- `sequence`: WESR metrics (default)
- `classification`: Classification accuracy

**Tag types:**
- `by_type`: Separate evaluation for discrete `[tag]` and continuous `<tag>` tags
- `combined`: Combined evaluation of all tags

**Configuration options:**
- `big`: Enable tag category aggregation


## Output Format

### Evaluation Results

The evaluation outputs a markdown table with per-tag and aggregate metrics:

```
| Tag | Precision | Recall | F1 |
|-----|-----------|--------|----|
|breathing|0.025|0.021|0.023|
|chuckle|0.157|0.492|0.238|
|clear_throat|0.526|0.690|0.597|
|cough|0.761|0.545|0.635|
|...|...|...|...|...|...|...|
|whispering|0.856|0.700|0.771|
|Micro|0.711|0.716|0.713|
|Macro|0.412|0.415|0.380|
```

### Results by Tag Type

When using `tag_type="by_type"`, results are separated into three sections:

1. **Discrete tags [tag]**: Metrics for bracket-style tags
2. **Continuous tags <tag>**: Metrics for angle-bracket tags
3. **Combined**: Overall metrics for all tags

### Metrics Explained

- **Micro**: Aggregated across all instances
- **Macro**: Averaged across all tag types

## Example

```python
from eval.eval import evaluate, EvaluationConfig, format_output_by_tag_type

# Evaluate with all tags
config = EvaluationConfig(include_tags=None, big=False)
results = evaluate(
    true_path=["eval/wesr_bench.jsonl"],
    pred_path="predictions.jsonl",
    config=config,
    eval_type="sequence",
    tag_type="by_type"
)

print(format_output_by_tag_type(results))
```

## Citation

If you find WESR helpful in your research, please cite our paper:

```
@misc{yang2026wesrscalingevaluatingwordlevel,
      title={WESR: Scaling and Evaluating Word-level Event-Speech Recognition}, 
      author={Chenchen Yang and Kexin Huang and Liwei Fan and Qian Tu and Botian Jiang and Dong Zhang and Linqi Yin and Shimin Li and Zhaoye Fei and Qinyuan Cheng and Xipeng Qiu},
      year={2026},
      eprint={2601.04508},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04508}, 
}
```
