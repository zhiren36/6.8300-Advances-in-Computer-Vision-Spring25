# 6.8300 Problem Set 4

> Differentiable Rendering

This problem set will cover

- Neural networks as representations of signals
- Sphere tracing and differentiable volume rendering
- Neural radiance fields

Please refer to the READMEs in `problem1/`, `problem2/`, `problem3/` for implementation details and submission instructions.

## Grading

### Points breakdown
- Problem 1 [Total 40 points]
    - Multiple choice: 10%
    - Implementing SIRENs: 20%
    - Benchmarking experiment: 10%
- Problem 2 [Total of 30 points]
    - Sphere tracing: 15%
    - Volumetric rendering: 15%
- Problem 3 [Total of 30 points]
    - Implementation and understanding of the rendering: 25%
    - NeRF results images: 5%

Submission is via Gradescope like for previous problem sets.

### Grading mechanism

For each problem, look in the description to see how its graded. The grades are all in `[0, 1]`. We take a weighted average with the weights above to get the final grade in terms of all the problems:

```python
import numpy as np
weights_names = ["p1_mc", "p1_siren", "p2_siren", "p2_sphere", "p2_vol", "p3_impl", "p3_nerf"]
weights = np.array([0.10, 0.20, 0.10, 0.15, 0.15, 0.25, 0.05])  # weights for each problem
grades = np.array([0.90, 0.85, 1.05, 0.80, 0.75, 0.90, 0.85])   # example grades in same order - this is determined by the grader per-problem
final_score = np.dot(weights, grades).item()                    # compute weighted sum
print(f"Final grade: {final_score:.4%}")
```