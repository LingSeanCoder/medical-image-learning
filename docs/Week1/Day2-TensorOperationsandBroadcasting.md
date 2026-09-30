# Day 2 — Tensor Operations and Broadcasting

## 1. Learning Objectives

In this lesson, I learned how to:

* Perform basic tensor operations.
* Calculate `mean`, `sum`, and `norm`.
* Understand matrix multiplication using `@`.
* Understand the shape rule for matrix multiplication.
* Understand broadcasting in PyTorch.
* Calculate global and channel-wise mean/std.
* Normalize image tensors.
* Apply these concepts to image data and WSI feature representations.

---

# 2. Basic Tensor Operations

Example:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([10.0, 20.0, 30.0])

print("x + y =", x + y)
print("x * y =", x * y)
print("mean(x) =", x.mean())
print("sum(x) =", x.sum())
print("norm(x) =", torch.norm(x))
```

### Element-wise operations

`x + y` and `x * y` operate on corresponding elements.

```text
x = [1, 2, 3]
y = [10, 20, 30]

x + y = [11, 22, 33]
x * y = [10, 40, 90]
```

Important:

```text
+  → element-wise addition
*  → element-wise multiplication
```

`*` is **not** matrix multiplication.

---

# 3. Mean, Sum, and Norm

### Mean

The mean is the average value:

```math
mean(x) = \frac{x_1+x_2+\cdots+x_n}{n}
```

For:

```text
x = [1, 2, 3]
```

```text
mean = 2
```

### Sum

The sum adds all elements:

```text
sum([1, 2, 3]) = 6
```

### Norm

The L2 norm measures the magnitude of a vector:

```math
\|x\|_2 = \sqrt{x_1^2+x_2^2+\cdots+x_n^2}
```

For:

```text
x = [1, 2, 3]
```

```math
\|x\|_2 = \sqrt{14} \approx 3.742
```

Norms are commonly used with vectors, embeddings, similarity, and optimization.

---

# 4. Matrix Multiplication

Example:

```python
A = torch.randn(3, 4)
B = torch.randn(4, 2)

C = A @ B

print("A:", A.shape)
print("B:", B.shape)
print("C:", C.shape)
```

Shapes:

```text
A → (3, 4)
B → (4, 2)
C → (3, 2)
```

The general rule is:

```text
(m, n) @ (n, p) = (m, p)
```

The two inner dimensions must match.

```text
(3, 4) @ (4, 2)
   ↑          ↑
   └── must be equal
```

The output keeps the outer dimensions:

```text
(3, 2)
```

---

# 5. How Matrix Multiplication Works

Each output element is calculated from:

> one row of the first matrix × one column of the second matrix.

Mathematically:

```math
C_{ij} = \sum_k A_{ik}B_{kj}
```

For example:

```text
A row = [a₁, a₂, a₃, a₄]

B column = [b₁, b₂, b₃, b₄]
```

Then:

```math
C_{ij}
=
a_1b_1+a_2b_2+a_3b_3+a_4b_4
```

---

# 6. Connection to Neural Networks

Matrix multiplication is a fundamental operation in neural networks.

A basic linear layer can be written as:

```math
z = Wx + b
```

where:

* `x` = input
* `W` = weight matrix
* `b` = bias
* `z` = output

The basic flow is:

```text
Input x
   ↓
W @ x
   ↓
+ bias b
   ↓
Output z
```

Therefore, understanding matrix multiplication is essential for understanding neural network layers.

---

# 7. Broadcasting

Broadcasting allows PyTorch to perform operations between tensors with compatible shapes without manually copying data.

Example:

```python
images = torch.randn(8, 3, 224, 224)

channel_mean = images.mean(
    dim=(0, 2, 3),
    keepdim=True
)

centered_images = images - channel_mean
```

Shapes:

```text
images
(8, 3, 224, 224)

channel_mean
(1, 3, 1, 1)

centered_images
(8, 3, 224, 224)
```

PyTorch automatically broadcasts:

```text
(1, 3, 1, 1)
```

across:

```text
(8, 3, 224, 224)
```

---

# 8. Why `dim=(0,2,3)`?

For an image batch:

```text
(B, C, H, W)
```

we want a separate mean for each channel.

Therefore, we calculate the mean across:

```text
B → dimension 0
H → dimension 2
W → dimension 3
```

while keeping:

```text
C → dimension 1
```

So:

```text
(8, 3, 224, 224)
        ↓
mean over B, H, W
        ↓
(1, 3, 1, 1)
```

For an RGB image, this gives three values:

```text
mean_R
mean_G
mean_B
```

---

# 9. Image Normalization

A common normalization formula is:

```math
x_{norm} = \frac{x-\mu}{\sigma}
```

where:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation
* `x_norm` = normalized value

After standard normalization, the data should have approximately:

```text
mean ≈ 0
std ≈ 1
```

Example:

```text
x = 80
mean = 60
std = 10
```

Then:

```math
x_{norm} = \frac{80-60}{10}=2
```

---

# 10. Global vs Channel-wise Normalization

## Global normalization

One mean and one standard deviation are calculated for the entire tensor.

```python
mean = images.mean()
std = images.std()

normalized = (images - mean) / std
```

There is only one:

```text
mean
std
```

for the whole dataset/tensor being normalized.

---

## Channel-wise normalization

For RGB images, mean and standard deviation can be calculated separately for each channel.

```python
mean = images.mean(
    dim=(0, 2, 3),
    keepdim=True
)

std = images.std(
    dim=(0, 2, 3),
    keepdim=True
)

normalized = (images - mean) / std
```

The resulting shapes are:

```text
mean → (1, 3, 1, 1)
std  → (1, 3, 1, 1)
```

This allows broadcasting to apply different statistics to:

```text
Red
Green
Blue
```

---

# 11. Exercise

## Grayscale Batch

Create a batch with shape:

```text
(10, 1, 28, 28)
```

This represents:

```text
10 grayscale images
1 channel
28 × 28 pixels
```

Calculate:

```python
mean = images.mean()
std = images.std()
```

Then normalize:

```python
normalized = (images - mean) / std
```

Check:

```python
print(normalized.mean())
print(normalized.std())
```

Expected:

```text
mean ≈ 0
std ≈ 1
```

---

# 12. RGB Batch

Create:

```text
(8, 3, 32, 32)
```

This represents:

```text
8 RGB images
3 channels
32 × 32 pixels
```

Calculate mean and standard deviation separately for each channel:

```python
mean = images.mean(
    dim=(0, 2, 3),
    keepdim=True
)

std = images.std(
    dim=(0, 2, 3),
    keepdim=True
)
```

Normalize:

```python
normalized = (images - mean) / std
```

Check the mean and standard deviation of each channel.

Expected:

```text
Mean ≈ [0, 0, 0]
Std  ≈ [1, 1, 1]
```

---

# 13. Complete Exercise Code

```python
import torch

# ============================================================
# 1. Grayscale images: (10, 1, 28, 28)
# ============================================================
gray_images = torch.randn(10, 1, 28, 28)

# Calculate global mean and standard deviation
gray_mean = gray_images.mean()
gray_std = gray_images.std()

# Normalize the entire tensor
gray_norm = (gray_images - gray_mean) / gray_std

print("=== Grayscale ===")
print("Shape:", gray_images.shape)
print("Before | mean:", gray_images.mean().item(),
      "| std:", gray_images.std().item())
print("After  | mean:", gray_norm.mean().item(),
      "| std:", gray_norm.std().item())


# ============================================================
# 2. RGB images: (8, 3, 32, 32)
# ============================================================
rgb_images = torch.randn(8, 3, 32, 32)

# Calculate mean/std separately for each channel
# dim=(0,2,3): batch, height, width
rgb_mean = rgb_images.mean(
    dim=(0, 2, 3),
    keepdim=True
)

rgb_std = rgb_images.std(
    dim=(0, 2, 3),
    keepdim=True
)

# Broadcasting applies each channel's statistics
rgb_norm = (rgb_images - rgb_mean) / rgb_std

print("\n=== RGB ===")
print("Shape:", rgb_images.shape)

print("Before:")
print("Mean per channel:", rgb_mean.flatten())
print("Std per channel:", rgb_std.flatten())

# Check statistics after normalization
after_mean = rgb_norm.mean(dim=(0, 2, 3))
after_std = rgb_norm.std(dim=(0, 2, 3))

print("After:")
print("Mean per channel:", after_mean)
print("Std per channel:", after_std)
```

---

# 14. Key Takeaways

The main concepts learned in Day 2 are:

### Element-wise operations

```text
x + y
x - y
x * y
x / y
```

Operations are performed element by element.

### Matrix multiplication

```text
(m, n) @ (n, p) → (m, p)
```

The inner dimensions must match.

### Image batch

```text
(B, C, H, W)
```

### Channel-wise statistics

```text
(B, C, H, W)
        ↓
mean(dim=(0,2,3), keepdim=True)
        ↓
(1, C, 1, 1)
```

### Broadcasting

```text
(B, C, H, W)
-
(1, C, 1, 1)
```

PyTorch automatically applies the channel-specific values to every image and every pixel.

### Normalization

```math
x_{norm} = \frac{x-\mu}{\sigma}
```

Usually produces:

```text
mean ≈ 0
std ≈ 1
```

---

# 15. Connection to Medical Imaging and WSI

These tensor operations are directly relevant to medical imaging.

For image-based models:

```text
(B, C, H, W)
```

represents a batch of images.

For WSI feature-based models:

```text
(S, P, D)
```

can represent:

```text
S = number of slides
P = patches per slide
D = feature dimension
```

For example:

```text
(16, 500, 768)
```

means:

```text
16 slides
500 patches per slide
768 features per patch
```

Understanding tensor shape, matrix multiplication, broadcasting, and normalization is therefore an important foundation for later work with **CNNs, Vision Transformers, WSI, and Multiple Instance Learning (MIL)**.
