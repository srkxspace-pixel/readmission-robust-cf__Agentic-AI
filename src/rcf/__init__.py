# torch must load before sklearn/xgboost on this Windows setup, else c10.dll fails (WinError 1114).
import torch  # noqa: F401
