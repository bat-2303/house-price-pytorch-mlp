import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna

class HousePriceNet(nn.Module):
    # Kiến trúc giờ đây nhận tham số từ bên ngoài truyền vào
    def __init__(self, input_size, l1_units, l2_units, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, l1_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Chống học vẹt
            
            nn.Linear(l1_units, l2_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(l2_units, 1)
        )

    def forward(self, x):
        return self.network(x)

def main():
    print("Đang chuẩn bị dữ liệu...")
    PROJECT_DIR = Path.cwd()
    DATA_DIR = PROJECT_DIR / "house-price"
    
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X = train.drop(columns=["SalePrice", "Id"])
    y = train["SalePrice"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = X_train.select_dtypes(include="number").columns
    categorical_cols = X_train.select_dtypes(include=["object", "str", "category", "bool"]).columns

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_val_processed = X_val_processed.toarray()

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)

    y_train_tensor = torch.tensor(np.log1p(y_train.to_numpy()), dtype=torch.float32).reshape(-1, 1)
    y_val_tensor = torch.tensor(np.log1p(y_val.to_numpy()), dtype=torch.float32).reshape(-1, 1)

   
    def objective(trial):
        l1_units = trial.suggest_categorical("l1_units", [64, 128, 256])
        l2_units = trial.suggest_categorical("l2_units", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = HousePriceNet(X_train_tensor.shape[1], l1_units, l2_units, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Chạy thử 150 vòng cho mỗi cấu hình để chấm điểm nhanh
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor)

        # Trả về điểm thi thử để Optuna biết cấu hình này tốt hay tệ
        return val_loss.item()

    
    print("\nBắt đầu dò tìm thông số bằng Optuna (20 lượt)...")
    study = optuna.create_study(direction="minimize") # Mục tiêu là làm Loss càng nhỏ càng tốt
    study.optimize(objective, n_trials=20)

    print("\nTHÔNG SỐ TỐT NHẤT TÌM ĐƯỢC:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"   - {key}: {value}")

    
    print("\nHuấn luyện mô hình cuối cùng (500 epoch) với thông số xịn...")
    final_model = HousePriceNet(
        X_train_tensor.shape[1], 
        best_params["l1_units"], 
        best_params["l2_units"], 
        best_params["dropout_rate"]
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])

    for epoch in range(500):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()

    # --- DỰ ĐOÁN & XUẤT FILE ---
    print("\nĐang xuất file kết quả...")
    X_test = test.drop(columns=["Id"])
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()
    
    final_model.eval()
    with torch.no_grad():
        test_preds_log = final_model(torch.tensor(X_test_processed, dtype=torch.float32)).numpy()

    test_preds_usd = np.clip(np.expm1(test_preds_log), a_min=0, a_max=None)
    
    pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": test_preds_usd.squeeze()
    }).to_csv(DATA_DIR / "submission.csv", index=False)
    
    print("Đã xuất file thành công! Sẵn sàng nộp Kaggle.")

if __name__ == "__main__":
    main()