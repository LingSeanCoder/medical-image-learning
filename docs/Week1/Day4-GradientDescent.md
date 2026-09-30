# Day 4 — Gradient Descent từ con số 0

## 1. Mục tiêu

Sau buổi này, cần hiểu được toàn bộ quá trình:

```text
Data
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
Update parameters
 ↓
Repeat
```

Ta chưa dùng Neural Network.

Thay vào đó, cho máy học một đường thẳng:

$$
\hat{y}=wx+b
$$

Dữ liệu được tạo từ:

$$
y=2x+1+\text{noise}
$$

Mục tiêu:

$$
w\approx2,\qquad b\approx1
$$

---

# 2. Ý tưởng của bài toán

Ta có dữ liệu:

```text
x → y
```

Ví dụ:

```text
x = 1  →  y ≈ 3
x = 2  →  y ≈ 5
x = 3  →  y ≈ 7
```

Model ban đầu chưa biết đường thẳng đúng.

Nó bắt đầu với `w` và `b` ngẫu nhiên:

```text
w = random
b = random
```

Sau đó lặp lại:

```text
1. Predict
2. Calculate Loss
3. Calculate Gradient
4. Update w, b
5. Reset Gradient
```

---

# 3. Cell 7 — Tạo dữ liệu

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

x = torch.linspace(-3, 3, 100).unsqueeze(1)

noise = 0.4 * torch.randn_like(x)

y = 2.0 * x + 1.0 + noise

plt.figure(figsize=(6, 4))
plt.scatter(x.numpy(), y.numpy(), s=15)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic regression data")
plt.grid(True)
plt.show()
```

## Giải thích

### `torch.manual_seed(42)`

```python
torch.manual_seed(42)
```

Cố định random seed để mỗi lần chạy lại đều tạo ra cùng dữ liệu ngẫu nhiên.

Điều này giúp kết quả **reproducible**.

---

### `torch.linspace()`

```python
x = torch.linspace(-3, 3, 100)
```

Tạo 100 giá trị cách đều từ:

$$
-3\rightarrow3
$$

Sau:

```python
.unsqueeze(1)
```

shape trở thành:

```text
(100, 1)
```

Thay vì:

```text
(100,)
```

---

### Tạo noise

```python
noise = 0.4 * torch.randn_like(x)
```

`torch.randn_like(x)` tạo noise theo phân phối chuẩn.

Nhân với `0.4` làm noise nhỏ hơn:

$$
\text{noise}=0.4\epsilon
$$

---

### Tạo `y`

```python
y = 2.0 * x + 1.0 + noise
```

Tương ứng:

$$
y=2x+1+\text{noise}
$$

Đường thẳng thật có:

```text
w = 2
b = 1
```

---

# 4. Cell 8 — Tự train `w` và `b`

```python
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.05
loss_history = []

for step in range(500):

    # Forward pass
    y_pred = w * x + b

    # Mean Squared Error
    loss = ((y_pred - y) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset gradients
    w.grad.zero_()
    b.grad.zero_()

    loss_history.append(loss.item())

    if step % 100 == 0:
        print(
            f"step={step:03d} "
            f"loss={loss.item():.4f} "
            f"w={w.item():.4f} "
            f"b={b.item():.4f}"
        )
```

Kết quả sẽ có dạng:

```text
step=000 loss=...
step=100 loss=...
step=200 loss=...
step=300 loss=...
step=400 loss=...
```

Cuối cùng:

```text
w ≈ 2
b ≈ 1
```

Giá trị chính xác có thể hơi khác vì dữ liệu có noise.

---

# 5. Forward Pass

```python
y_pred = w * x + b
```

Đây là model prediction:

$$
\boxed{\hat{y}=wx+b}
$$

Ví dụ:

```text
w = 2
b = 1
x = 3
```

thì:

$$
\hat{y}=2(3)+1=7
$$

---

# 6. Loss

```python
loss = ((y_pred - y) ** 2).mean()
```

Đây là **Mean Squared Error (MSE)**:

$$
\boxed{
\mathcal{L}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
}
$$

Loss đo mức độ dự đoán của model khác với dữ liệu thật bao nhiêu.

### Loss nhỏ

```text
Prediction ≈ True value
```

→ Model dự đoán tốt.

### Loss lớn

```text
Prediction far from True value
```

→ Model dự đoán chưa tốt.

---

# 7. Backward Pass

```python
loss.backward()
```

PyTorch tự động tính:

$$
\frac{\partial\mathcal{L}}{\partial w}
$$

và:

$$
\frac{\partial\mathcal{L}}{\partial b}
$$

Đây chính là phần **Autograd** đã học ở Day 3.

Có thể hình dung:

```text
Loss
 ↓ backward()
Gradients
 ↓
∂L/∂w
∂L/∂b
```

---

# 8. Gradient Descent

Sau khi có gradient, ta cập nhật parameter:

```python
w -= learning_rate * w.grad
b -= learning_rate * b.grad
```

Tương ứng:

$$
\boxed{
w
\leftarrow
w-\eta\frac{\partial\mathcal{L}}{\partial w}
}
$$

$$
\boxed{
b
\leftarrow
b-\eta\frac{\partial\mathcal{L}}{\partial b}
}
$$

Trong đó:

* \(w,b\): parameters
* \(\mathcal{L}\): loss
* \(\eta\): learning rate
* \(\frac{\partial\mathcal{L}}{\partial w}\): gradient theo \(w\)
* \(\frac{\partial\mathcal{L}}{\partial b}\): gradient theo \(b\)

---

# 9. Vì sao lại trừ gradient?

Gradient chỉ hướng mà Loss **tăng nhanh nhất**.

Ta muốn Loss giảm.

Vì vậy đi theo hướng ngược gradient:

```text
Gradient direction
       ↑
       |
       | Loss increases
       |
       ●
       |
       | ← move opposite gradient
       |
       ↓
    lower Loss
```

Do đó:

$$
\text{new parameter}
=
\text{old parameter}
-
\text{learning rate}\times\text{gradient}
$$

---

# 10. Learning Rate

Learning rate quyết định **mỗi bước model di chuyển bao xa**.

Trong bài:

```python
learning_rate = 0.05
```

### `learning_rate = 0.001`

```text
Bước rất nhỏ
→ học chậm
→ có thể cần nhiều iteration
```

### `learning_rate = 0.05`

```text
Bước hợp lý
→ Loss giảm
→ model hội tụ
```

### `learning_rate = 1.0`

```text
Bước rất lớn
→ có thể vượt qua minimum
→ dao động
→ hoặc diverge
```

Không phải lúc nào `1.0` cũng chắc chắn diverge; điều này còn phụ thuộc vào dữ liệu, loss và scale của bài toán.

---

# 11. `torch.no_grad()`

Khi update:

```python
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
```

Ta **không muốn PyTorch ghi lại phép update này vào computational graph**.

Đây là bước thay đổi parameter bằng tay, không phải phép tính cần Autograd tiếp tục theo dõi.

Nếu không dùng `torch.no_grad()`, việc update parameter có thể làm computational graph bị mở rộng không cần thiết và gây vấn đề khi train.

### Hiểu đơn giản

```text
Forward
→ cần Autograd
→ cần track

Parameter update
→ không cần Autograd
→ no_grad()
```

---

# 12. Vì sao phải xóa Gradient?

```python
w.grad.zero_()
b.grad.zero_()
```

Đây là điểm rất quan trọng.

PyTorch mặc định **accumulate gradient**.

Ví dụ:

```text
Step 1:
gradient = 2

Step 2:
gradient mới = 3
```

Nếu không reset:

```text
gradient = 2 + 3 = 5
```

thay vì chỉ sử dụng gradient của Step 2.

Vì vậy sau mỗi update:

```python
w.grad.zero_()
b.grad.zero_()
```

để chuẩn bị cho iteration tiếp theo.

---

# 13. Toàn bộ Training Loop

Đây là phần quan trọng nhất của Day 4:

```text
                 ┌──────────────┐
                 │     Data     │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │   Forward    │
                 │ y_pred=wx+b  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │     Loss     │
                 │     MSE      │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │   backward() │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  Gradients   │
                 │ ∂L/∂w, ∂L/∂b│
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │    Update    │
                 │ w,b -= ηgrad │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Reset Grad   │
                 └──────┬───────┘
                        ↓
                     Repeat
```

---

# 14. Bài tập 1 — Vẽ Loss

Sau khi training:

```python
plt.figure(figsize=(6, 4))

plt.plot(loss_history)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.grid(True)
plt.show()
```

### Quan sát

Loss thường có dạng:

```text
Loss
 ↑
 |\
 | \
 |  \
 |   \______
 |
 +----------------→ Step
```

Tức là Loss giảm dần rồi tiến tới một giá trị ổn định.

---

# 15. Bài tập 2 — Vẽ dữ liệu và đường dự đoán

```python
with torch.no_grad():
    y_pred = w * x + b

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
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
```

Ta sẽ thấy:

```text
●  ●
  ●   ●
───────  ← learned line
 ● ●
```

Đường dự đoán sẽ nằm gần các điểm dữ liệu.

---

# 16. Bài tập 3 — Thử Learning Rate

Chạy lại training với:

```python
learning_rate = 0.001
```

sau đó:

```python
learning_rate = 0.05
```

và:

```python
learning_rate = 1.0
```

So sánh:

| Learning Rate | Hiện tượng cần quan sát         |
| ------------: | ------------------------------- |
|       `0.001` | Update rất nhỏ, học chậm        |
|        `0.05` | Thường hội tụ tốt trong bài này |
|         `1.0` | Có thể dao động hoặc diverge    |

### Quan trọng

Không nên ghi:

> `lr = 1.0` luôn luôn diverge.

Đúng hơn:

> Learning rate quá lớn **có thể** khiến quá trình học dao động hoặc diverge.

---

# 17. Bài tập 4 — Tự giải thích

## Câu 1. Vì sao `loss.backward()` không tự cập nhật `w,b`?

`backward()` chỉ **tính gradient**.

Nó không quyết định cách update parameter.

```text
loss.backward()
      ↓
calculate gradients
      ↓
w.grad
b.grad
```

Sau đó ta mới update:

```python
w -= learning_rate * w.grad
b -= learning_rate * b.grad
```

Trong PyTorch thực tế, phần update thường được giao cho **optimizer** như SGD hoặc Adam.

---

## Câu 2. Vì sao cần `torch.no_grad()`?

Vì update parameter không cần được ghi vào computational graph.

Ta chỉ muốn:

```text
gradient → parameter update
```

chứ không muốn Autograd tiếp tục theo dõi chính thao tác update đó.

---

## Câu 3. Vì sao phải xóa gradient?

Vì PyTorch mặc định cộng dồn gradient.

Nếu không reset:

```text
Gradient step 1
      +
Gradient step 2
      +
Gradient step 3
      ...
```

→ gradient bị tích lũy ngoài ý muốn.

Do đó cần:

```python
w.grad.zero_()
b.grad.zero_()
```

sau mỗi update.

---

# 18. Từ Gradient Descent đến Neural Network

Bài hôm nay chỉ có:

$$
\hat{y}=wx+b
$$

Nhưng neural network cũng hoạt động theo cùng nguyên lý:

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
Gradients
 ↓
Optimizer
 ↓
Update parameters
```

Điểm khác là neural network có **rất nhiều parameters**:

```text
w1, w2, w3, ...
b1, b2, b3, ...
```

thay vì chỉ:

```text
w, b
```

---

# 19. Connection với Day 3

### Day 3 — Autograd

Ta học:

```text
x
 ↓
function
 ↓
y
 ↓
backward()
 ↓
gradient
```

### Day 4 — Gradient Descent

Ta dùng gradient để **học**:

```text
Parameters
 ↓
Forward
 ↓
Loss
 ↓
backward()
 ↓
Gradient
 ↓
Update Parameters
 ↓
Better Parameters
```

Vì vậy:

$$
\boxed{
\text{Autograd}
\rightarrow
\text{Gradient}
\rightarrow
\text{Gradient Descent}
\rightarrow
\text{Learning}
}
$$

---

# 20. Key Takeaways

### 1. Model

$$
\boxed{\hat{y}=wx+b}
$$

### 2. Loss

$$
\boxed{
\mathcal{L}
=
\frac{1}{N}
\sum_i(\hat{y}_i-y_i)^2
}
$$

### 3. Backward

```python
loss.backward()
```

Tính:

$$
\frac{\partial L}{\partial w},
\qquad
\frac{\partial L}{\partial b}
$$

### 4. Update

$$
\boxed{
w\leftarrow w-\eta\frac{\partial L}{\partial w}
}
$$

$$
\boxed{
b\leftarrow b-\eta\frac{\partial L}{\partial b}
}
$$

### 5. Reset

```python
w.grad.zero_()
b.grad.zero_()
```

để tránh gradient bị cộng dồn.

---

# 21. Công thức tư duy quan trọng nhất

Khi nhìn một training loop PyTorch, hãy nhận ra ngay:

```text
FORWARD
y_pred = model(x)

LOSS
loss = criterion(y_pred, y)

BACKWARD
loss.backward()

UPDATE
optimizer.step()

RESET
optimizer.zero_grad()
```

Day 4 chính là phiên bản **làm thủ công** của quy trình trên.

Sau này khi dùng:

```python
optimizer = torch.optim.SGD(...)
```

hoặc:

```python
optimizer = torch.optim.Adam(...)
```

ta không cần tự viết công thức update `w` và `b` nữa.

Nhưng bản chất toán học phía sau vẫn là:

$$
\boxed{
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
}
$$
