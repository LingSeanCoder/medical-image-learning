# Week 01 Review — Tensor, Autograd and Gradient Descent

## 1. Tensor khác NumPy array ở đâu?

Tensor và NumPy array đều là cấu trúc dùng để lưu dữ liệu nhiều chiều, nhưng Tensor được thiết kế đặc biệt cho Machine Learning và Deep Learning.

Một số điểm khác nhau quan trọng:

* Tensor có thể chạy trên **GPU**, còn NumPy array chủ yếu chạy trên CPU.
* Tensor tích hợp trực tiếp với **Autograd** của PyTorch để tính gradient.
* Tensor là kiểu dữ liệu chính được sử dụng trong các model PyTorch.
* NumPy mạnh về numerical computing và xử lý dữ liệu, nhưng không trực tiếp tham gia vào computational graph của PyTorch.

Ví dụ:

```python
import torch
import numpy as np

x_np = np.array([1, 2, 3])
x_tensor = torch.tensor([1, 2, 3])
```

Có thể hiểu đơn giản:

```text
NumPy array
    ↓
Numerical computing

PyTorch Tensor
    ↓
Numerical computing
    +
GPU
    +
Autograd
    +
Deep Learning
```

Tensor không phải là "tốt hơn NumPy trong mọi trường hợp". Chúng phục vụ các mục đích khác nhau và có thể chuyển đổi qua lại trong nhiều trường hợp.

---

## 2. `requires_grad=True` làm gì?

`requires_grad=True` yêu cầu PyTorch theo dõi các phép tính liên quan đến tensor đó để sau này có thể tính gradient.

Ví dụ:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2

y.backward()

print(x.grad)
```

Kết quả:

```text
tensor(4.)
```

Vì:

```text
y = x²

dy/dx = 2x

x = 2

dy/dx = 4
```

Điều này rất quan trọng khi train neural network vì model cần biết:

> Thay đổi parameter một chút thì loss thay đổi như thế nào?

Gradient cung cấp thông tin đó.

---

## 3. `loss.backward()` làm gì?

`loss.backward()` thực hiện **backpropagation**.

Nó tính gradient của loss đối với các parameter đang được theo dõi bởi Autograd.

Ví dụ model có:

```text
weight
bias
```

Sau:

```python
loss.backward()
```

PyTorch sẽ tính:

```text
∂Loss/∂weight
∂Loss/∂bias
```

và lưu các gradient này trong:

```python
parameter.grad
```

Ví dụ:

```python
loss.backward()

print(model.weight.grad)
print(model.bias.grad)
```

Có thể hình dung:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
backward()
  ↓
Gradient
  ↓
weight.grad
bias.grad
```

`backward()` **không trực tiếp cập nhật parameter**.

Nó chỉ tính gradient.

---

## 4. `optimizer.zero_grad()` làm gì?

PyTorch mặc định **tích lũy gradient**.

Điều đó có nghĩa là nếu gọi:

```python
loss.backward()
```

nhiều lần mà không xóa gradient cũ, gradient mới sẽ được cộng vào gradient trước đó.

Vì vậy trong training loop thường có:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Thứ tự:

```text
zero_grad()
    ↓
Xóa gradient cũ

loss.backward()
    ↓
Tính gradient mới

optimizer.step()
    ↓
Cập nhật parameter
```

Nếu quên `zero_grad()`, gradient có thể bị cộng dồn qua nhiều iteration và làm quá trình training sai.

---

## 5. `optimizer.step()` làm gì?

`optimizer.step()` sử dụng gradient vừa tính để **cập nhật các parameter của model**.

Ví dụ với Gradient Descent:

```text
parameter_new
=
parameter_old
-
learning_rate × gradient
```

Ví dụ:

```text
weight = 5.0
gradient = 2.0
learning rate = 0.1
```

thì:

```text
new weight
= 5.0 - 0.1 × 2.0
= 4.8
```

Trong PyTorch:

```python
loss.backward()
optimizer.step()
```

`backward()`:

```text
Tính gradient
```

`step()`:

```text
Dùng gradient để cập nhật parameter
```

Đây là hai việc khác nhau.

---

## 6. Learning rate quá nhỏ / quá lớn gây gì?

Learning rate quyết định **mức độ lớn của mỗi bước cập nhật parameter**.

### Learning rate quá nhỏ

Ví dụ:

```python
lr = 0.000001
```

Model có thể học rất chậm.

Loss giảm nhưng cần rất nhiều iteration để tiến tới vùng tốt.

```text
Loss
│\
│ \
│  \
│   \____
│
└──────────── Epoch
```

Vấn đề chính:

* Training rất lâu.
* Có thể chưa hội tụ trong số epoch hiện tại.

### Learning rate quá lớn

Ví dụ:

```python
lr = 10
```

Model có thể cập nhật quá mạnh.

Thay vì tiến dần về minimum, nó có thể:

```text
nhảy qua lại
   ↓
dao động
   ↓
loss tăng
   ↓
thậm chí divergence
```

Loss có thể trông như:

```text
Loss
│    /\    /\
│   /  \  /  \
│__/    \/    \__
│
└──────────────── Epoch
```

Vì vậy learning rate là một hyperparameter rất quan trọng.

---

## 7. Loss và metric khác nhau thế nào?

### Loss

Loss là đại lượng được sử dụng trong quá trình training để model tối ưu.

Ví dụ:

```python
criterion = nn.MSELoss()
```

Sau đó:

```python
loss = criterion(prediction, target)
```

Model cố gắng làm:

```text
Loss ↓
```

Loss phải phù hợp với mục tiêu tối ưu và thường cần khả vi để backpropagation.

### Metric

Metric dùng để **đánh giá performance** của model.

Ví dụ:

Classification:

```text
Accuracy
Precision
Recall
F1-score
AUC
```

Regression:

```text
MAE
MSE
RMSE
R²
```

Ví dụ:

```text
Training:
    MSE Loss ↓

Evaluation:
    Accuracy ↑
```

Loss và metric đôi khi có thể giống nhau, nhưng không bắt buộc phải giống.

Điểm quan trọng:

> Loss dùng để optimization; metric dùng để đánh giá model.

---

## 8. Batch là gì?

Batch là một **nhóm sample được đưa vào model cùng một lần**.

Ví dụ dataset có:

```text
10,000 images
```

Nếu:

```text
batch_size = 32
```

thì mỗi iteration model xử lý:

```text
32 images
```

thay vì toàn bộ 10,000 images.

Ví dụ với image classification:

```text
Dataset
10,000 images
       ↓
Batch 1 → 32 images
Batch 2 → 32 images
Batch 3 → 32 images
...
```

Một epoch nghĩa là model đã đi qua toàn bộ training dataset một lần.

Nếu có:

```text
10,000 samples
batch_size = 32
```

thì số iteration trong một epoch xấp xỉ:

```text
10,000 / 32 ≈ 313
```

Batch giúp:

* giảm memory usage;
* tận dụng GPU tốt;
* cập nhật model nhiều lần trong một epoch.

---

## 9. Shape `(16, 3, 224, 224)` nghĩa là gì?

Đây là một batch gồm **16 ảnh RGB**, mỗi ảnh có kích thước `224 × 224`.

Ta có:

```text
(16, 3, 224, 224)
 │   │   │    │
 │   │   │    └── Width
 │   │   └─────── Height
 │   └─────────── Channels
 └─────────────── Batch size
```

Cụ thể:

```text
16   → 16 images
3    → RGB channels
224  → Height
224  → Width
```

Vì vậy:

```text
(16, 3, 224, 224)
```

có nghĩa:

> 16 ảnh RGB, mỗi ảnh có kích thước 224 × 224 pixels.

Đây là format thường gặp trong PyTorch Computer Vision:

```text
(B, C, H, W)
```

Trong đó:

```text
B = Batch
C = Channel
H = Height
W = Width
```

Ví dụ:

```text
(32, 3, 224, 224)
```

là:

```text
32 RGB images
```

---

## 10. Trong WSI sau này, `(16, 500, 768)` có thể nghĩa là gì?

Trong bài toán Whole Slide Imaging (WSI), một cách hiểu hợp lý là:

```text
(16, 500, 768)
 │   │    │
 │   │    └── Feature dimension
 │   └─────── Number of patches
 └─────────── Number of slides
```

Có thể hiểu:

```text
16 slides
    ↓
mỗi slide có 500 patches
    ↓
mỗi patch được biểu diễn bằng 768 features
```

Tức là:

```text
16 WSI
×
500 patches / WSI
×
768-dimensional feature / patch
```

Ví dụ một slide:

```text
Slide 1
│
├── Patch 1  → [768 features]
├── Patch 2  → [768 features]
├── Patch 3  → [768 features]
│
└── Patch 500 → [768 features]
```

Khi gom 16 slide lại:

```text
(16, 500, 768)
```

Đây là dạng dữ liệu rất phù hợp để đi tới các bài toán như **Multiple Instance Learning (MIL)**.

Ý tưởng quan trọng là:

```text
WSI
 ↓
chia thành nhiều patches
 ↓
Feature extractor
 ↓
mỗi patch → vector 768 chiều
 ↓
500 patch vectors
 ↓
MIL / aggregation
 ↓
Slide-level prediction
```

Khác với image classification thông thường:

```text
Image
↓
CNN
↓
Prediction
```

WSI thường có cấu trúc:

```text
Whole Slide
↓
Many patches
↓
Features of patches
↓
Aggregate information
↓
Slide-level prediction
```

Vì vậy việc hiểu shape từ sớm rất quan trọng khi sau này làm Medical AI.

---

# Week 01 Mental Model

Toàn bộ tuần 1 có thể gom lại thành một pipeline:

```text
Tensor
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
loss.backward()
  ↓
Gradient
  ↓
optimizer.step()
  ↓
Updated Parameters
  ↓
Prediction tốt hơn
```

Training loop cơ bản:

```python
for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(x)

    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()
```

Trong đó:

```text
zero_grad()
    → xóa gradient cũ

model(x)
    → forward pass

criterion(...)
    → tính loss

backward()
    → tính gradient

step()
    → cập nhật parameter
```

Đây là nền tảng để hiểu các model phức tạp hơn sau này như:

```text
CNN
ResNet
DenseNet
Vision Transformer
MIL
Medical Imaging models
```

---

# Week 01 Checklist

* [x] Tensor
* [x] Tensor shape
* [x] Tensor operations
* [x] Broadcasting
* [x] Matrix multiplication
* [x] `requires_grad=True`
* [x] Autograd
* [x] `loss.backward()`
* [x] Gradient Descent
* [x] `nn.Module`
* [x] `nn.Linear`
* [x] `MSELoss`
* [x] Optimizer
* [x] `optimizer.zero_grad()`
* [x] `optimizer.step()`
* [x] Learning rate
* [x] Batch
* [x] Train / Validation
* [x] Loss curve
* [x] Prediction curve
* [x] Basic WSI tensor representation

---

# Git Commit

Sau khi tạo và kiểm tra:

```text
docs/week01_review.md
```

Mở terminal tại root của project rồi chạy:

```bash
git add .
git commit -m "Complete week 1 tensor autograd and gradient descent"
git push
```

Kiểm tra trước khi push:

```bash
git status
```

Nếu mọi thứ đã được commit thành công, có thể thấy:

```text
nothing to commit, working tree clean
```

Sau đó kiểm tra repository trên GitHub để chắc chắn file:

```text
docs/
└── week01_review.md
```

đã xuất hiện.
