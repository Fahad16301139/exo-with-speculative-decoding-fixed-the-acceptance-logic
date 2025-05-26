# Speculative Decoding for Exo

This module implements speculative decoding for the Exo distributed inference system. Speculative decoding is a technique that can speed up autoregressive text generation by using a smaller, faster "draft" model to generate candidate tokens that are then verified by the larger "target" model.

## How It Works

1. **Draft Generation**: A smaller, faster model generates K candidate tokens
2. **Parallel Verification**: The target model evaluates all K+1 positions in parallel
3. **Acceptance/Rejection**: Tokens are accepted or rejected using speculative sampling
4. **Speedup**: Multiple tokens can be generated in roughly the time of one target model forward pass

## Algorithm

The implementation follows Algorithm 1 from the [Speculative Decoding paper](https://arxiv.org/pdf/2211.17192.pdf):

```
For each draft token x_i:
  1. Compute p_target(x_i) and p_draft(x_i)
  2. Accept with probability min(1, p_target(x_i) / p_draft(x_i))
  3. If rejected, sample from adjusted distribution: max(0, p_target - p_draft)
```

## Usage

### Command Line

Enable speculative decoding when starting Exo:

```bash
# Basic usage with auto-selected draft model
exo --enable-speculative --draft-tokens 4

# Specify custom draft model
exo --enable-speculative --draft-tokens 4 --draft-model llama-3.2-1b

# Advanced configuration
exo --enable-speculative \
    --draft-tokens 6 \
    --draft-model llama-3.2-3b \
    --speculative-threshold 0.8 \
    --max-speculation-depth 8
```

### API Usage

Use speculative decoding via the ChatGPT-compatible API:

```python
import requests

response = requests.post("http://localhost:52415/v1/chat/completions", json={
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "speculative": {
        "enabled": True,
        "draft_tokens": 4,
        "draft_model": "llama-3.2-3b"
    }
})
```

## Configuration

### Parameters

- **`enabled`**: Enable/disable speculative decoding
- **`draft_tokens`**: Number of tokens to generate with draft model (default: 4)
- **`draft_model`**: Draft model to use (auto-selected if not specified)
- **`acceptance_threshold`**: Minimum acceptance probability (default: 0.8)
- **`max_speculation_depth`**: Maximum speculation depth (default: 8)
- **`adaptive_speculation`**: Enable adaptive speculation (default: True)

### Model Pairs

The system includes predefined draft model mappings:

| Target Model | Draft Model | Draft Tokens |
|--------------|-------------|--------------|
| llama-3.1-70b | llama-3.2-3b | 4 |
| llama-3.1-8b | llama-3.2-1b | 3 |
| qwen-2.5-72b | qwen-2.5-7b | 4 |
| deepseek-v3 | deepseek-r1-distill-qwen-7b | 4 |

## Performance

Speculative decoding can provide significant speedups, especially for:

- **Large target models** (70B+ parameters)
- **Long sequences** (more opportunities for speculation)
- **High-quality draft models** (better acceptance rates)

Typical speedups range from 1.5x to 3x depending on the model pair and sequence characteristics.

## Metrics

The system tracks several metrics:

- **Acceptance Rate**: Percentage of draft tokens accepted
- **Speedup Ratio**: Estimated speedup over normal inference
- **Draft Time**: Time spent generating draft tokens
- **Verification Time**: Time spent verifying with target model

Access metrics via the inference engine:

```python
if hasattr(node.inference_engine, 'get_metrics'):
    metrics = node.inference_engine.get_metrics()
    print(f"Acceptance rate: {metrics['acceptance_rate']:.2f}")
    print(f"Estimated speedup: {metrics['estimated_speedup']:.2f}x")
```

## Implementation Details

### Architecture

- **`SpeculativeInferenceEngine`**: Main wrapper that coordinates draft and target models
- **`SpeculativeConfig`**: Configuration class for all parameters
- **Model Integration**: Seamless integration with existing MLX and Tinygrad engines

### Distributed Support

The implementation works with Exo's distributed architecture:

- Draft and target models can run on different devices
- Automatic load balancing and coordination
- Metrics aggregation across nodes

### Error Handling

Robust fallback mechanisms ensure reliability:

- Falls back to normal inference if draft model fails
- Graceful handling of model loading errors
- Automatic retry logic for transient failures

## Examples

See `examples/speculative_example.py` for a complete example comparing normal and speculative inference performance.

## Limitations

- **Memory Usage**: Requires loading both draft and target models
- **Model Compatibility**: Draft and target models should be from similar families
- **Sequence Length**: Benefits decrease for very short sequences
- **Temperature**: Works best with lower temperature settings

## Future Improvements

- **Adaptive Draft Tokens**: Dynamically adjust based on acceptance rate
- **Multi-Draft Models**: Use multiple draft models for different contexts
- **Batched Speculation**: Speculate across multiple requests simultaneously
- **Tree-based Speculation**: Explore multiple candidate paths 