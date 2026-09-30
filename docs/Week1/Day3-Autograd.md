# Day 3 — Autograd: “Máy tính Gradient”

## 1. Mục tiêu

Sau buổi này, cần hiểu được:

1. `requires_grad=True` dùng để làm gì.
2. `backward()` hoạt động như thế nào.
3. `.grad` chứa thông tin gì.
4. Mối liên hệ giữa **Autograd → Gradient → Backpropagation → Neural Network**.

---

# 2. Autograd là gì?

**Autograd** là hệ thống tự động tính đạo hàm của PyTorch.

Ví dụ:

$$
y=x^2+3x+1
$$

Ta có:

$$
\frac{dy}{dx}=2x+3
$$

Thay vì tự tính đạo hàm bằng tay, PyTorch có thể tự tính:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 1

y.backward()

print(x.grad)
```

Kết quả:

```text
tensor(7.)
```

Vì:

$$
\frac{dy}{dx}\bigg|_{x=2}
=2(2)+3
=7
$$

---

# 3. Cell 6 — Đạo hàm đơn giản

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 1

y.backward()

print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())
```

### Expected output

```text
x = 2.0
y = 11.0
dy/dx = 7.0
```

---

# 4. `requires_grad=True`

```python
x = torch.tensor(2.0, requires_grad=True)
```

`requires_grad=True` yêu cầu PyTorch:

> Theo dõi các phép toán liên quan đến `x` để sau này có thể tính gradient.

Nếu không bật:

```python
x = torch.tensor(2.0)
```

thì PyTorch không tự động tính gradient cho `x`.

### Cần nhớ

```text
requires_grad=True
        ↓
PyTorch theo dõi phép tính
        ↓
x tham gia vào computational graph
        ↓
có thể tính gradient bằng backward()
```

---

# 5. Computational Graph

Khi chạy:

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 1
```

PyTorch xây dựng một **computational graph** mô tả cách `y` được tạo ra từ `x`.

Có thể hình dung đơn giản:

```text
        x
       / \
     x²   3x
       \ /
        +
        |
      + 1
        |
        y
```

Graph này giúp PyTorch biết cần áp dụng chain rule như thế nào khi tính gradient.

---

# 6. `y.backward()`

```python
y.backward()
```

Có thể hiểu đơn giản là:

> Bắt đầu từ `y` và tính ngược gradient về các tensor cần gradient.

Đây là quá trình **backward pass**.

PyTorch sử dụng **chain rule** để tính các đạo hàm.

Ví dụ:

$$
x \rightarrow u \rightarrow y
$$

thì:

$$
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
$$

Đây chính là **chain rule**.

---

# 7. `x.grad`

Sau khi chạy:

```python
y.backward()
```

ta có:

```python
x.grad
```

`x.grad` chứa:

$$
\frac{\partial y}{\partial x}
$$

Trong ví dụ:

```python
print(x.grad)
```

kết quả:

```text
tensor(7.)
```

Nghĩa là:

$$
\frac{dy}{dx}=7
$$

---

# 8. Tại sao Gradient quan trọng?

Gradient cho biết:

> Nếu thay đổi một biến một chút, kết quả sẽ thay đổi theo hướng và mức độ nào.

Trong Machine Learning, ta thường có:

$$
\text{Input}
\rightarrow
\text{Model}
\rightarrow
\text{Prediction}
\rightarrow
\text{Loss}
$$

Sau đó:

```text
Loss
 ↓
backward()
 ↓
Gradients
 ↓
Update parameters
 ↓
Model tốt hơn
```

Ví dụ một neural network có parameter \(w\).

Ta cần:

$$
\frac{\partial L}{\partial w}
$$

trong đó:

* \(L\): Loss
* \(w\): model parameter
* \(\frac{\partial L}{\partial w}\): gradient của Loss theo parameter

Gradient được dùng để cập nhật parameter.

Một dạng đơn giản của Gradient Descent:

$$
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}
$$

Trong đó:

* \(w\): parameter
* \(L\): Loss
* \(\eta\): learning rate

---

# 9. Autograd và Neural Network

Trong neural network, có rất nhiều parameter:

```text
W1, b1
W2, b2
W3, b3
...
```

Model tính:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Prediction
  ↓
Loss
```

Sau đó:

```python
loss.backward()
```

PyTorch tính gradient cho các parameter:

```text
∂Loss/∂W1
∂Loss/∂b1
∂Loss/∂W2
∂Loss/∂b2
...
```

Optimizer sau đó sử dụng các gradient này để cập nhật parameter.

Vì vậy:

```text
Autograd
   ↓
Gradient
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Update model
```

Đây là một trong những cơ chế cốt lõi để **train neural network**.

---

# 10. Bài tập — Kiểm tra Autograd

Cho:

$$
f(x)=3x^3-2x^2+5
$$

tại:

$$
x=2
$$

## Bước 1 — Tự tính đạo hàm

$$
f'(x)=9x^2-4x
$$

Tại \(x=2\):

$$
f'(2)=9(2^2)-4(2)
$$

$$
=36-8
$$

$$
=\boxed{28}
$$

---

## Bước 2 — Kiểm tra bằng PyTorch

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

f = 3*x**3 - 2*x**2 + 5

f.backward()

print("x =", x.item())
print("f(x) =", f.item())
print("df/dx =", x.grad.item())
```

### Expected output

```text
x = 2.0
f(x) = 25.0
df/dx = 28.0
```

Ta thấy:

$$
\boxed{\text{PyTorch gradient}=28}
$$

trùng với kết quả tính bằng tay.

---

# 11. Luồng hoạt động cần nhớ

```text
Create tensor
      ↓
requires_grad=True
      ↓
Perform operations
      ↓
Computational Graph
      ↓
Calculate output / Loss
      ↓
backward()
      ↓
Gradient
      ↓
tensor.grad
```

---

# 12. Key Takeaways

| Thành phần           | Ý nghĩa                                  |
| -------------------- | ---------------------------------------- |
| `requires_grad=True` | Theo dõi phép tính để tính gradient      |
| Computational Graph  | Lưu quan hệ giữa các phép toán           |
| `backward()`         | Tính gradient theo chiều ngược           |
| `.grad`              | Gradient của tensor                      |
| Chain Rule           | Quy tắc toán học dùng để tính gradient   |
| Autograd             | Cơ chế tự động tính gradient của PyTorch |
| Backpropagation      | Lan truyền gradient ngược qua network    |

### Công thức quan trọng

$$
\boxed{\frac{dy}{dx}}
$$

là gradient của \(y\) theo \(x\).

Trong Machine Learning:

$$
\boxed{
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}
}
$$

Gradient là thứ giúp model biết **nên thay đổi parameter theo hướng nào để giảm Loss**.

---

# 13. Connection với những buổi tiếp theo

Day 1:

```text
Tensor
```

Day 2:

```text
Tensor Operations
+
Broadcasting
```

Day 3:

```text
Tensor
  ↓
Operations
  ↓
Autograd
  ↓
Gradient
```

Từ đây có thể xây dựng:

```text
Neural Network
      ↓
Forward Pass
      ↓
Loss
      ↓
Backward Pass
      ↓
Gradients
      ↓
Optimizer
      ↓
Updated Parameters
```

Đây là nền tảng để hiểu cách **PyTorch train model**, bao gồm cả các model Deep Learning dùng trong Computer Vision và Medical Imaging.
