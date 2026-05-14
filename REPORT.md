# CSC4005 Lab 3 Report - UrbanSound8K with 1D-CNN

## 1. Thong tin sinh vien

- Ho ten: Nguyen Duy Thuan
- Ma sinh vien: 1771040024
- Lop: CSC4005
- Link GitHub repo: cap nhat theo link GitHub Classroom
- W&B: da chay offline, can sync cac run trong thu muc `wandb/`

Run offline:

```text
wandb/offline-run-20260514_091110-bklv1ims  - MFCC baseline
wandb/offline-run-20260514_091254-shu77u10  - log-mel extension
wandb/offline-run-20260514_091503-un1cg0lh  - raw waveform extension
```

Lenh sync W&B:

```bash
wandb sync wandb/offline-run-20260514_091110-bklv1ims
wandb sync wandb/offline-run-20260514_091254-shu77u10
wandb sync wandb/offline-run-20260514_091503-un1cg0lh
```

## 2. Muc tieu thi nghiem

Muc tieu cua lab la xay dung pipeline phan loai 10 lop am thanh moi truong trong UrbanSound8K. Audio duoc dua ve cung sample rate, cung do dai, trich xuat dac trung theo thoi gian, sau do huan luyen 1D-CNN va phan tich ket qua bang learning curves, confusion matrix va W&B.

## 3. Du lieu va tien xu ly

- Dataset: UrbanSound8K
- So lop: 10
- Cac lop: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`
- Train folds: 1-8
- Validation fold: 9
- Test fold: 10
- Split sau khi gioi han mau/lop: train 1200, validation 463, test 465

| Thanh phan | Gia tri |
|---|---|
| Sample rate | 16000 Hz |
| Duration | 4.0 giay |
| Feature baseline | MFCC |
| n_mfcc / n_mels | 40 / 64 |
| n_fft | 1024 |
| hop_length | 512 |
| Augmentation | time/frequency mask cho MFCC/log-mel; shift + noise cho raw waveform |

Can dua audio ve cung sample rate va cung do dai vi 1D-CNN nhan tensor co kich thuoc on dinh theo batch. Neu moi file co tan so lay mau hoac do dai khac nhau, model se kho hoc pattern nhat quan va DataLoader khong the ghep batch truc tiep.

## 4. Mo hinh 1D-CNN

Baseline su dung `Feature1DCNN`:

```text
Input [batch, feature_channels, time_frames]
-> Conv1D + BatchNorm + ReLU + MaxPool
-> Conv1D + BatchNorm + ReLU + MaxPool
-> Conv1D + BatchNorm + ReLU + MaxPool
-> AdaptiveAvgPool1D
-> Dropout
-> Linear classifier 10 lop
```

| Thanh phan | Gia tri |
|---|---|
| model_name | `mfcc_1dcnn` |
| hidden_channels | [64, 128, 128] |
| dropout | 0.35 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| batch size | 32 |
| epochs | 12 |
| patience | 4 |
| trainable params | 137,930 |

Input MFCC co shape gan dung `[batch, 40, T]`, trong do 40 la so kenh dac trung va `T` la so frame thoi gian. Kernel Conv1D truot theo chieu thoi gian de hoc cac pattern cuc bo trong chuoi dac trung audio.

## 5. Ket qua thuc nghiem

### 5.1 Baseline MFCC + 1D-CNN

| Metric | Gia tri |
|---|---:|
| Best validation accuracy | 0.6091 |
| Test accuracy | 0.5097 |
| Best validation loss | 1.2760 |
| Test loss | 1.3418 |
| Average epoch time | 6.07 s |
| Trainable parameters | 137,930 |

Artifact:

- `outputs/1771040024_mfcc_1dcnn_baseline/metrics.json`
- `outputs/1771040024_mfcc_1dcnn_baseline/history.csv`
- `outputs/1771040024_mfcc_1dcnn_baseline/curves.png`
- `outputs/1771040024_mfcc_1dcnn_baseline/confusion_matrix.png`

Learning curves cho thay train accuracy tang rat cao den 0.9875, trong khi validation accuracy dat dinh 0.6091 roi dao dong. Dieu nay cho thay co dau hieu overfitting: model hoc tot tap train nhung kha nang tong quat tren fold validation/test con han che.

### 5.2 Confusion matrix baseline

Mot so nham lan lon cua baseline MFCC:

- `air_conditioner` bi du doan thanh `engine_idling`: 44/50 mau.
- `jackhammer` bi du doan thanh `engine_idling`: 21/50 mau.
- `drilling` bi du doan thanh `engine_idling`: 16/50 mau.
- `children_playing` bi du doan thanh `street_music`: 16/50 mau.
- `siren` bi du doan thanh `children_playing`: 11/50 mau.

Nguyen nhan hop ly la cac lop co am nen lien tuc, tieng may hoac pho thi co pho tan so gan nhau. Rieng `children_playing`, `street_music` va `siren` co the cung xuat hien trong moi truong duong pho nhieu nhieu nen dac trung MFCC bi chong lap.

## 6. So sanh cac pipeline

| Pipeline | Feature/Input | Best val acc | Test accuracy | Avg epoch time | Nhan xet |
|---|---|---:|---:|---:|---|
| Baseline | MFCC + 1D-CNN | 0.6091 | 0.5097 | 6.07 s | Chay on nhung overfit ro, nham nhieu vao `engine_idling`. |
| Extension 1 | log-mel + 1D-CNN | 0.6631 | 0.6065 | 6.63 s | Tot nhat trong 3 run, giu nhieu thong tin pho hon MFCC. |
| Extension 2 | raw waveform + 1D-CNN | 0.5832 | 0.6022 | 33.11 s | Test gan log-mel nhung train cham hon nhieu, can nhieu du lieu/thoi gian hon. |

Huong tot nhat trong cac thi nghiem nay la log-mel + 1D-CNN vi co test accuracy cao nhat va thoi gian train gan baseline MFCC. Raw waveform dat ket qua test gan log-mel nhung epoch cham hon khoang 5 lan, nen khong phu hop bang log-mel neu muc tieu la pipeline on dinh tren CPU/laptop.

## 7. Tra loi cau hoi thao luan

1. Dung 1D-CNN thay vi MLP vi audio feature la chuoi theo thoi gian. Conv1D khai thac duoc pattern cuc bo va chia se trong so theo truc thoi gian, trong khi MLP de mat cau truc lien tiep.
2. Kernel 1D truot theo chieu thoi gian cua chuoi MFCC/log-mel. Cac kenh dau vao la cac he so MFCC hoac mel bins.
3. MFCC giup model hoc de hon raw waveform vi da tom tat thong tin pho am thanh, giam so chieu va bot bat model tu hoc dac trung tan so tu tin hieu tho.
4. Han che hien tai la du lieu train bi gioi han theo lop, model con overfit, va baseline MFCC nham nhieu cac lop tieng may/duong pho.
5. Co the cai thien bang log-mel, augmentation tot hon, tang du lieu moi lop, train lau hon co regularization, dung cross-validation theo 10 folds, hoac phan tich cac mau sai de dieu chinh tien xu ly.

## 8. Ket luan

Pipeline da doc dung UrbanSound8K, chuan hoa audio, trich xuat MFCC/log-mel/raw waveform va train duoc 1D-CNN. Baseline MFCC dat test accuracy 50.97%, log-mel dat 60.65% va raw waveform dat 60.22%. Ket qua cho thay log-mel la lua chon tot nhat trong dieu kien chay hien tai vi can thoi gian gan MFCC nhung accuracy cao hon ro ret. Confusion matrix cho thay cac lop am thanh co nen may moc hoac moi truong duong pho de bi nham lan, dac biet `air_conditioner`, `engine_idling`, `drilling`, `jackhammer`, `children_playing` va `street_music`.
