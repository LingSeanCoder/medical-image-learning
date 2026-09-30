# Tensor Basics — What I Learned

## 1. What is a Tensor?

A **tensor** is a data structure used in PyTorch to store numerical data.

A tensor can have different numbers of dimensions:

* **0D** → Scalar
* **1D** → Vector
* **2D** → Matrix
* **3D** → Image
* **4D** → Batch of images
* Higher dimensions can represent more complex data structures.

---

## 2. Tensor Shape

The `shape` describes the size of the tensor along each dimension.

Example:

```python
x = torch.randn(16, 3, 224, 224)
print(x.shape)
```

Output:

```text
torch.Size([16, 3, 224, 224])
```

This means the tensor has 4 dimensions with sizes:

```text
16 × 3 × 224 × 224
```

`ndim` tells us how many dimensions the tensor has.

---

## 3. Scalar — `()`

A scalar contains only one value.

```python
scalar = torch.tensor(7.0)
```

Shape:

```text
()
```

Meaning:

```text
0 dimensions
1 value
```

Example:

```text
7.0
```

---

## 4. Vector — `(3,)`

A vector is a 1D tensor.

```python
vector = torch.tensor([1.0, 2.0, 3.0])
```

Shape:

```text
(3,)
```

Meaning:

```text
1 dimension
3 elements
```

Example:

```text
[1.0, 2.0, 3.0]
```

---

## 5. Matrix — `(2, 3)`

A matrix is a 2D tensor.

```python
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
```

Shape:

```text
(2, 3)
```

Meaning:

```text
2 rows
3 columns
```

So the tensor contains:

```text
2 × 3 = 6 values
```

---

## 6. Image Tensor — `(C, H, W)`

Images are commonly represented in PyTorch as:

```text
(C, H, W)
```

where:

* `C` = Number of channels
* `H` = Height
* `W` = Width

### Grayscale image

```python
image = torch.randn(1, 28, 28)
```

Shape:

```text
(1, 28, 28)
```

Meaning:

```text
1 channel
28 pixels high
28 pixels wide
```

A grayscale image has one channel because each pixel has one intensity value.

### RGB image

```python
rgb_image = torch.randn(3, 224, 224)
```

Shape:

```text
(3, 224, 224)
```

Meaning:

```text
3 channels
224 pixels high
224 pixels wide
```

The three channels are:

```text
R = Red
G = Green
B = Blue
```

---

## 7. Batch of Images — `(B, C, H, W)`

When processing multiple images at the same time, PyTorch commonly uses:

```text
(B, C, H, W)
```

where:

* `B` = Batch size
* `C` = Number of channels
* `H` = Height
* `W` = Width

Example:

```python
batch = torch.randn(16, 3, 224, 224)
```

Shape:

```text
(16, 3, 224, 224)
```

Meaning:

```text
16 images
3 channels per image
224 pixels high
224 pixels wide
```

---

## 8. FashionMNIST Example

FashionMNIST contains grayscale images.

One image:

```text
(1, 28, 28)
```

A batch of images:

```text
(B, 1, 28, 28)
```

For example, a batch of 32 images:

```text
(32, 1, 28, 28)
```

Meaning:

```text
32 images
1 channel
28 × 28 pixels
```

---

## 9. WSI Patch Features

In Whole Slide Imaging (WSI), a large slide is usually divided into many smaller image patches.

For example:

```text
Whole Slide Image
        ↓
    Patch 1
    Patch 2
    Patch 3
       ...
    Patch 1000
```

A patch can have an image shape such as:

```text
(3, 224, 224)
```

A feature extraction model can convert each patch into a feature vector.

For example:

```text
Image patch
(3, 224, 224)
      ↓
Feature extractor
      ↓
Feature vector
(768,)
```

Therefore, 1000 patches can be represented as:

```text
(1000, 768)
```

Meaning:

```text
1000 patches
768 features for each patch
```

---

## 10. WSI Features for Multiple Slides

For multiple slides, we can use a 3D tensor:

```text
(16, 500, 768)
```

Meaning:

```text
16 slides
500 patches per slide
768 features per patch
```

The dimensions can be interpreted as:

```text
(S, P, D)
```

where:

* `S` = Number of slides
* `P` = Number of patches per slide
* `D` = Feature/embedding dimension

So:

```text
(16, 500, 768)
```

means:

> 16 slides; each slide has up to 500 patches; each patch is represented by a 768-dimensional feature vector.

This type of tensor is commonly encountered when working with **Multiple Instance Learning (MIL)** for WSI analysis.

---

## 11. Important Shape Patterns

The most important patterns learned in this lesson are:

| Shape          | Meaning                               |
| -------------- | ------------------------------------- |
| `()`           | Scalar                                |
| `(D,)`         | Vector with `D` elements              |
| `(H, W)`       | 2D matrix/image                       |
| `(C, H, W)`    | One image                             |
| `(B, C, H, W)` | Batch of images                       |
| `(N, D)`       | `N` feature vectors with `D` features |
| `(S, P, D)`    | Slides × patches × features           |

---

## 12. Key Takeaways

1. **Tensor** is the basic data structure used by PyTorch for numerical data.
2. **Shape** tells us the size of the tensor in each dimension.
3. **`ndim`** tells us the number of dimensions.
4. For images, PyTorch commonly uses:

```text
(C, H, W)
```

5. For image batches:

```text
(B, C, H, W)
```

6. Grayscale images usually have:

```text
C = 1
```

7. RGB images usually have:

```text
C = 3
```

8. WSI data can be represented as:

```text
(S, P, D)
```

where each slide contains multiple patches and each patch is represented by a feature vector.

9. Understanding tensor shapes is essential for working with **Deep Learning, Medical Imaging, WSI, and MIL**.
