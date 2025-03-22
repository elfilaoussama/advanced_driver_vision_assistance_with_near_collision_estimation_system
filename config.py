import torch

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'detr_model_path': 'facebook/detr-resnet-101',
    'glpn_model_path': 'vinvino02/glpn-kitti',
    'lstm_model_path': 'data/models/pretrained_lstm.pth',
    'lstm_scaler_path': 'data/models/lstm_scaler.pkl',
    'xgboost_path': 'data/models/xgboost_model.json',
    'xgboost_scaler_path': 'data/models/scaler.joblib',
    'numerical_cols_path': 'data/models/numerical_columns.joblib'
}
