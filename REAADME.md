# 🏡 Dự đoán Giá Nhà (House Prices Prediction) - PyTorch Deep Learning

Dự án học máy End-to-End dự đoán giá nhà dựa trên tập dữ liệu [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) của Kaggle.

Mô hình sử dụng mạng nơ-ron đa tầng (Multi-Layer Perceptron - MLP) được xây dựng hoàn toàn bằng **PyTorch**, kết hợp với luồng tiền xử lý dữ liệu tự động của **Scikit-learn** và hệ thống dò tìm siêu tham số của **Optuna**.

## 🚀 Kỹ thuật & Công nghệ áp dụng

- **Tiền xử lý dữ liệu (Data Preprocessing):**
  - Sử dụng `Scikit-learn Pipeline` và `ColumnTransformer` để tự động hóa luồng làm sạch dữ liệu.
  - Điền giá trị thiếu bằng `SimpleImputer` (dùng Median cho biến số, Constant cho biến phân loại).
  - Chuẩn hóa tỷ lệ (Standardization) bằng `StandardScaler` chống bùng nổ trọng số.
  - Mã hóa biến phân loại bằng `OneHotEncoder`.
- **Mạng Nơ-ron & Toán học (PyTorch):**
  - Kiến trúc MLP linh hoạt với các hàm kích hoạt `ReLU` và kỹ thuật `Dropout` để chống học vẹt (Overfitting).
  - **Xử lý độ lệch phân phối (Skewness):** Biến đổi biến mục tiêu (Giá nhà) bằng Log Transform (`np.log1p`) trước khi đưa vào hàm suy hao MSE, và khôi phục giá trị thực bằng `np.expm1` ở đầu ra nhằm tối ưu hóa theo hệ số chấm điểm RMSLE của Kaggle.
  - Thuật toán tối ưu hóa `Adam Optimizer`.
- **Tối ưu siêu tham số (Hyperparameter Tuning):**
  - Tích hợp `Optuna` để tự động dò tìm cấu trúc mạng tối ưu (số lượng nơ-ron, tỷ lệ dropout) và tốc độ học (Learning Rate) theo thang đo logarit.

## 📈 Kết quả

- **Mô hình cơ sở (MLP mặc định):** Kaggle Score đạt `0.17309`.
- **Mô hình tối ưu (MLP + Optuna + Dropout):** Kaggle Score đạt `0.17402`.

## 📂 Cấu trúc dự án

```text
house-price-pytorch-mlp/
├── house-price/                # Thư mục chứa dataset Kaggle (cần tải về)
│   ├── train.csv               # Tập huấn luyện
│   ├── test.csv                # Tập kiểm thử
│   └── submission.csv          # File kết quả nộp bài tự động sinh ra
├── house_price_prediction.py   # Mã nguồn chính (Pipeline, Optuna, Training loop)
├── requirements.txt            # Danh sách thư viện Python
└── README.md                   # Tài liệu dự án
```
