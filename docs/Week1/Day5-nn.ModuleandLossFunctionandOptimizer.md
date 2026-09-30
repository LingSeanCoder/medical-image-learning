# Day 5 — `nn.Module`, Loss Function và Optimizer

## 1. Mục tiêu

Sau Day 5, cần hiểu:

* `nn.Module` là gì và tại sao dùng `nn.Linear`.
* `model.parameters()` là gì.
* Loss function dùng để làm gì.
* `nn.MSELoss()` hoạt động như thế nào.
* Optimizer là gì.
* `optimizer.zero_grad()`, `loss.backward()` và `optimizer.step()` khác nhau thế nào.
* SGD và Adam khác nhau ở mức trực giác.
* Tại sao optimizer và learning rate phải được báo cáo trong paper.
* Hiểu workflow training chuẩn của PyTorch.

---

# 2. Từ Day 4 sang Day 5

Ở Day 4, ta tự thực hiện toàn bộ quá trình:

```text
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
Tự update w, b
  ↓
Reset gradient
```

Ở Day 5, PyTorch cung cấp các abstraction chuẩn:

```text
nn.Module
  ↓
Prediction
  ↓
Loss Function
  ↓
zero_grad()
  ↓
backward()
  ↓
optimizer.step()
```

Điểm quan trọng:

> Day 5 không thay đổi bản chất của Gradient Descent. PyTorch chỉ đóng gói các phần ta phải tự viết ở Day 4.

---

# 3. `nn.Module`

## 3.1. Model ở Day 4

Bài toán của chúng ta:

$$
\hat{y}=wx+b
$$

Ở Day 4, ta tự tạo:

```python
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
```

Sau đó tự update:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

---

## 3.2. Model chuẩn của PyTorch

PyTorch cung cấp:

```python
model = nn.Linear(1, 1)
```

`nn.Linear(1, 1)` tương ứng với:

```text
Input:  1 feature
Output: 1 feature
```

Về toán học:

$$
y=Wx+b
$$

Trong trường hợp này:

```text
W = weight
b = bias
```

Có thể kiểm tra:

```python
print(model.weight)
print(model.bias)
```

`model.weight` và `model.bias` chính là những parameters mà model cần học.

---

# 4. `nn.Module` là gì?

`nn.Module` là base class được sử dụng để xây dựng và quản lý các model/neural network trong PyTorch.

Ví dụ:

```python
nn.Linear(...)
nn.Conv2d(...)
nn.ReLU(...)
nn.Dropout(...)
```

Các layer/model này được xây dựng dựa trên hệ thống `nn.Module`.

Một neural network có thể được xây như:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

Ví dụ:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
```

Toàn bộ network này có thể được quản lý như một `nn.Module`.

---

# 5. `model.parameters()`

Khi tạo:

```python
model = nn.Linear(1, 1)
```

PyTorch tự tạo các learnable parameters:

```text
weight
bias
```

Khi viết:

```python
model.parameters()
```

PyTorch trả về các parameters mà optimizer cần cập nhật.

Do đó:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)
```

có nghĩa:

> Optimizer sẽ cập nhật tất cả parameters có thể học của model.

Ta không cần tự viết:

```python
w -= ...
b -= ...
```

nữa.

---

# 6. Loss Function

Ở Day 4, ta tự viết:

```python
loss = ((y_pred - y) ** 2).mean()
```

Đây là Mean Squared Error (MSE).

PyTorch cung cấp sẵn:

```python
criterion = nn.MSELoss()
```

Sau đó:

```python
loss = criterion(y_pred, y)
```

## Công thức MSE

$$
L=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

Trong đó:

* \(N\): số lượng samples.
* \(y_i\): giá trị thật.
* \(\hat{y}_i\): prediction của model.
* \(L\): loss.

Loss càng nhỏ thì prediction càng gần giá trị thật.

---

# 7. Tại sao dùng `criterion`?

Ta có:

```python
criterion = nn.MSELoss()
```

Sau đó:

```python
loss = criterion(y_pred, y)
```

Có thể hiểu:

```text
criterion
    ↓
quy tắc tính lỗi
    ↓
y_pred + y
    ↓
loss
```

Điều này giúp dễ dàng thay đổi loss function.

Ví dụ:

### Regression

```python
nn.MSELoss()
```

### Binary Classification

```python
nn.BCEWithLogitsLoss()
```

### Multi-class Classification

```python
nn.CrossEntropyLoss()
```

Loss được chọn dựa trên loại bài toán.

---

# 8. Optimizer

Optimizer chịu trách nhiệm cập nhật parameters của model dựa trên gradient.

Ở Day 4, ta tự viết:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

Ở Day 5, ta dùng:

```python
optimizer.step()
```

Ví dụ:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)
```

SGD = Stochastic Gradient Descent.

Về mặt cơ bản:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
$$

Trong đó:

* \(\theta\): model parameters.
* \(\eta\): learning rate.
* \(L\): loss.
* \(\nabla_\theta L\): gradient của loss theo parameters.

---

# 9. `optimizer.zero_grad()`

Đây là kiến thức liên quan trực tiếp đến Autograd ở Day 3.

Sau:

```python
loss.backward()
```

gradient được lưu trong:

```python
parameter.grad
```

PyTorch mặc định **accumulate gradients**.

Nghĩa là gradient mới có thể được cộng vào gradient cũ.

Vì vậy, trước mỗi lần backward thường cần:

```python
optimizer.zero_grad()
```

Workflow:

```text
Gradient cũ
    ↓
optimizer.zero_grad()
    ↓
Gradient = 0
    ↓
loss.backward()
    ↓
Gradient mới
```

---

# 10. `loss.backward()`

```python
loss.backward()
```

thực hiện Backpropagation.

Nó sử dụng computational graph và Chain Rule để tính gradient:

$$
\frac{\partial L}{\partial \theta}
$$

Ví dụ:

```text
Loss
 ↓
backward()
 ↓
∂L/∂weight
∂L/∂bias
```

`backward()` **chỉ tính gradient**.

Nó không tự cập nhật parameters.

---

# 11. `optimizer.step()`

Sau:

```python
loss.backward()
```

ta đã có gradient.

Ví dụ:

```text
model.weight.grad
model.bias.grad
```

Sau đó:

```python
optimizer.step()
```

optimizer sử dụng gradient để cập nhật parameters.

Với SGD cơ bản:

$$
\theta
\leftarrow
\theta-\eta\nabla_\theta L
$$

Workflow:

```text
loss.backward()
      ↓
calculate gradients
      ↓
optimizer.step()
      ↓
update parameters
```

---

# 12. Training Loop chuẩn của PyTorch

Training loop:

```python
for step in range(500):

    # Forward
    y_pred = model(x)

    # Loss
    loss = criterion(y_pred, y)

    # Reset gradients
    optimizer.zero_grad()

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()
```

Đây là một pattern cực kỳ quan trọng.

Sau này khi train neural network lớn hơn, workflow này vẫn gần như giữ nguyên.

---

# 13. Code hoàn chỉnh

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. Create synthetic data: y = 2x + 1 + noise
# ============================================================
torch.manual_seed(42)

x = torch.linspace(-3, 3, 100).unsqueeze(1)
noise = 0.4 * torch.randn_like(x)
y = 2.0 * x + 1.0 + noise


# ============================================================
# 2. Create PyTorch model
# ============================================================
# Linear model: y = wx + b
model = nn.Linear(1, 1)

# Mean Squared Error loss
criterion = nn.MSELoss()

# SGD optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)

loss_history = []


# ============================================================
# 3. Training loop
# ============================================================
for step in range(500):

    # Forward pass
    y_pred = model(x)

    # Calculate loss
    loss = criterion(y_pred, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Save loss
    loss_history.append(loss.item())

    if step % 100 == 0:
        weight = model.weight.item()
        bias = model.bias.item()

        print(
            f"step={step:03d} "
            f"loss={loss.item():.4f} "
            f"w={weight:.4f} "
            f"b={bias:.4f}"
        )


# ============================================================
# 4. Plot loss curve
# ============================================================
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("SGD Loss Curve")
plt.grid(True)
plt.show()


# ============================================================
# 5. Visualize learned line
# ============================================================
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(6, 4))

plt.scatter(
    x.numpy(),
    y.numpy(),
    s=15,
    label="Data"
)

plt.plot(
    x.numpy(),
    y_pred.numpy(),
    label="Prediction"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression with PyTorch")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 6. Print final parameters
# ============================================================
print("\nFinal model:")
print(f"weight = {model.weight.item():.4f}  (target ≈ 2)")
print(f"bias   = {model.bias.item():.4f}  (target ≈ 1)")
```

---

# 14. Map code với toán

| Toán                                    | PyTorch                 |
| --------------------------------------- | ----------------------- |
| \(\hat y=wx+b\)                         | `nn.Linear(1, 1)`       |
| MSE                                     | `nn.MSELoss()`          |
| \(\nabla_\theta L\)                     | `loss.backward()`       |
| Xóa gradient                            | `optimizer.zero_grad()` |
| \(\theta\leftarrow\theta-\eta\nabla L\) | `optimizer.step()`      |
| Learnable parameters                    | `model.parameters()`    |

---

# 15. So sánh Day 4 và Day 5

| Day 4 — Manual        | Day 5 — PyTorch         |
| --------------------- | ----------------------- |
| `w`, `b` tự tạo       | `nn.Linear()`           |
| Tự tính MSE           | `nn.MSELoss()`          |
| `loss.backward()`     | `loss.backward()`       |
| Tự update `w`, `b`    | `optimizer.step()`      |
| Tự reset gradient     | `optimizer.zero_grad()` |
| Tự quản lý parameters | `model.parameters()`    |

Bản chất vẫn giống nhau:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradient
   ↓
Update
   ↓
Repeat
```

---

# 16. Bài tập 1 — Thay SGD bằng Adam

Thay:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)
```

bằng:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.05
)
```

Adam = Adaptive Moment Estimation.

Adam không chỉ sử dụng gradient hiện tại mà còn duy trì thông tin về gradient qua các bước trước để điều chỉnh update cho từng parameter.

---

# 17. SGD vs Adam

## SGD

Trực giác:

```text
Gradient hiện tại
       ↓
Xác định hướng đi
       ↓
Bước theo hướng ngược gradient
```

Công thức cơ bản:

$$
\theta_{t+1}
=
\theta_t-\eta g_t
$$

Trong đó:

$$
g_t=\nabla_\theta L
$$

---

## Adam

Trực giác:

```text
Gradient hiện tại
       +
Thông tin gradient trước đó
       ↓
Điều chỉnh kích thước bước
       ↓
Update parameter
```

Adam có cơ chế adaptive learning rate cho từng parameter.

Không nên hiểu đơn giản rằng:

> Adam luôn tốt hơn SGD.

Optimizer nào phù hợp còn phụ thuộc vào:

* dataset;
* architecture;
* learning rate;
* batch size;
* loss;
* regularization;
* training setup.

---

# 18. Bài tập 2 — So sánh loss curve SGD và Adam

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. Create the same dataset
# ============================================================
torch.manual_seed(42)

x = torch.linspace(-3, 3, 100).unsqueeze(1)
noise = 0.4 * torch.randn_like(x)
y = 2.0 * x + 1.0 + noise


# ============================================================
# 2. Training function
# ============================================================
def train(optimizer_name, lr=0.05, steps=500):

    # Create a new model
    model = nn.Linear(1, 1)

    # MSE loss
    criterion = nn.MSELoss()

    # Choose optimizer
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

    loss_history = []

    # Training loop
    for step in range(steps):

        # Forward
        y_pred = model(x)

        # Calculate loss
        loss = criterion(y_pred, y)

        # Reset gradients
        optimizer.zero_grad()

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history


# ============================================================
# 3. Train with SGD
# ============================================================
sgd_model, sgd_losses = train(
    "SGD",
    lr=0.05
)


# ============================================================
# 4. Train with Adam
# ============================================================
adam_model, adam_losses = train(
    "Adam",
    lr=0.05
)


# ============================================================
# 5. Compare final results
# ============================================================
print("=== SGD ===")
print(f"weight = {sgd_model.weight.item():.4f}")
print(f"bias   = {sgd_model.bias.item():.4f}")
print(f"loss   = {sgd_losses[-1]:.4f}")

print("\n=== Adam ===")
print(f"weight = {adam_model.weight.item():.4f}")
print(f"bias   = {adam_model.bias.item():.4f}")
print(f"loss   = {adam_losses[-1]:.4f}")


# ============================================================
# 6. Compare loss curves
# ============================================================
plt.figure(figsize=(7, 4))

plt.plot(sgd_losses, label="SGD")
plt.plot(adam_losses, label="Adam")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("SGD vs Adam")
plt.legend()
plt.grid(True)
plt.show()
```

### Cần quan sát

1. Loss giảm nhanh hay chậm.
2. Đường loss của SGD và Adam khác nhau như thế nào.
3. Final loss của hai optimizer.
4. `weight` có tiến gần 2 không.
5. `bias` có tiến gần 1 không.

Không chỉ nhìn final loss. Loss curve cho biết **training dynamics** của optimizer.

---

# 19. Bài tập 3 — Thử learning rate

Thử:

```python
learning_rates = [
    0.001,
    0.01,
    0.05,
    0.1,
    1.0
]
```

Trực giác:

```text
Learning rate quá nhỏ
        ↓
Step rất nhỏ
        ↓
Learning chậm


Learning rate phù hợp
        ↓
Step vừa phải
        ↓
Converge tốt


Learning rate quá lớn
        ↓
Step quá mạnh
        ↓
Oscillation / Divergence
```

Learning rate là một trong những hyperparameter quan trọng nhất trong training.

---

# 20. Tại sao paper phải báo optimizer và learning rate?

Ví dụ paper ghi:

```text
Model: ResNet
Optimizer: Adam
Learning rate: 0.001
```

Những thông tin này quan trọng vì:

```text
Same architecture
       +
Different optimizer
       +
Different learning rate
       ↓
Different training dynamics
       ↓
Potentially different results
```

Nếu paper chỉ nói:

```text
We trained ResNet.
```

thì chưa đủ thông tin để tái hiện chính xác training procedure.

Khi đọc một paper về Deep Learning, cần chú ý:

```text
Architecture
Dataset
Preprocessing
Loss
Optimizer
Learning rate
Batch size
Epochs
Scheduler
Weight decay
Augmentation
```

Đặc biệt khi muốn **reproduce** kết quả của paper, các thông tin này rất quan trọng.

---

# 21. `model.train()` và `model.eval()`

Sau này sẽ thường gặp:

```python
model.train()
```

và:

```python
model.eval()
```

Với `nn.Linear` đơn giản ở Day 5, chúng gần như không tạo khác biệt.

Nhưng với neural network có:

* Dropout
* Batch Normalization

thì rất quan trọng.

Training:

```python
model.train()
```

Validation/Inference:

```python
model.eval()
```

Khi inference:

```python
model.eval()

with torch.no_grad():
    predictions = model(x)
```

Pattern này sẽ được sử dụng rất nhiều khi train Deep Learning model.

---

# 22. Connection với Deep Learning

Day 5:

```python
model = nn.Linear(1, 1)
```

chỉ là một model cực kỳ đơn giản.

Sau này có thể có:

```python
model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)
```

Hoặc CNN:

```text
Image
  ↓
Conv2d
  ↓
ReLU
  ↓
Pooling
  ↓
Conv2d
  ↓
...
  ↓
Linear
  ↓
Prediction
```

Nhưng training loop vẫn gần như:

```python
for epoch in range(epochs):

    model.train()

    y_pred = model(x)

    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Đây là workflow nền tảng của rất nhiều Deep Learning models.

---

# 23. Mental Model cần nhớ

```text
                 MODEL
                   │
              nn.Module
                   │
                   ▼
              Prediction
                   │
                   ▼
             LOSS FUNCTION
                   │
             nn.MSELoss()
                   │
                   ▼
            loss.backward()
                   │
                   ▼
              Gradients
                   │
                   ▼
          optimizer.step()
                   │
                   ▼
        Updated Parameters
                   │
                   └──────────► Repeat
```

---

# 24. Tóm tắt Day 5

### `nn.Module`

Quản lý architecture và learnable parameters của model.

### `nn.Linear`

Biểu diễn một linear layer:

$$
y=Wx+b
$$

### `nn.MSELoss`

Tính Mean Squared Error:

$$
L=
\frac{1}{N}
\sum_i(\hat y_i-y_i)^2
$$

### `optimizer.zero_grad()`

Xóa gradient cũ.

### `loss.backward()`

Tính gradient bằng Backpropagation.

### `optimizer.step()`

Cập nhật parameters dựa trên gradient.

### SGD

Update dựa trên gradient:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

### Adam

Sử dụng thông tin gradient qua nhiều bước và adaptive updates để điều chỉnh quá trình optimization.

---

# 25. Một câu phải nhớ

> **`nn.Module` quản lý model và parameters, loss function đo sai số, `backward()` tính gradient, còn optimizer dùng gradient để cập nhật parameters.**

Workflow cốt lõi:

```text
Model
  ↓
Prediction
  ↓
Loss
  ↓
zero_grad()
  ↓
backward()
  ↓
optimizer.step()
  ↓
Repeat
```
