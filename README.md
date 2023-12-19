# 用supervised訓練自走車避障
* aiTest : reference model
* car_dataCollect : 收集資料
* generateModel : 訓練
* 目前是將問題看成分類問題，0是前進、1是左轉、2是右轉
# 程式執行
* 先將unity環境內的trainingmanager改成非訓練模式 (https://github.com/alianlbj23/car_AI_supervised_ENV/tree/main)
* 跑蒐集資料用程式`python3 car_dataCollect.py`
* 創模型 `generateModel_MLP.py`
* 將unity那端模式改回訓練模式
* 執行`python3 inference_MLP.py`
