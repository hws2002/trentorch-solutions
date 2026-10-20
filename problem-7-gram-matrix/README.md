# Gram Matrix

Intermediate | problemset | maths-stats-for-ml | linear-algebra

Given a data matrix $X$ whose rows are samples and whose columns are features, compute the Gram matrix $X^\top X$ of its columns.

Implement `solve(X)`.

**Returns.** Return a symmetric NumPy array of shape $(d, d)$ for $d$ columns.

### Examples

**Example 1**

Input:

```python
solve([[1.0, 2.0], [3.0, 4.0]])
```

Output:

```text
[[10.0, 14.0], [14.0, 20.0]]
```

**Example 2**

Input:

```python
solve([[1.0, 0.0], [0.0, 1.0]])
```

Output:

```text
[[1.0, 0.0], [0.0, 1.0]]
```
