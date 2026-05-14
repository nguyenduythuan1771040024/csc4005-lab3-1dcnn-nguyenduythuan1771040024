# GitHub Classroom Guide

## 1. Clone repository

Clone repository assignment tu GitHub Classroom ve may ca nhan, sau do mo terminal tai thu muc repo.

```bash
git clone <github-classroom-repo-url>
cd <repo-folder>
```

## 2. Khong dua dataset len GitHub

Dataset UrbanSound8K co nhieu file audio nen khong nen commit vao repository. Khi chay bai lab, dat dataset ngoai repo hoac giai nen cuc bo tren may, sau do truyen duong dan bang tham so `--data_dir`.

Vi du:

```bash
python -m src.train --config configs/baseline_mfcc_1dcnn.json --data_dir UrbanSound8K
```

Neu trong thu muc lam viec co cac thu muc/file du lieu lon nhu `UrbanSound8K/`, `.cache/`, `wandb/` hoac model/output tam, can kiem tra `.gitignore` truoc khi commit.

## 3. Chay kiem tra truoc khi nop

Chay kiem tra cau truc repo:

```bash
python ci/check_structure.py
```

Chay smoke test de dam bao pipeline train hoat dong:

```bash
python ci/smoke_train.py
```

## 4. Nop bai

Truoc khi nop, can co toi thieu:

- code chay duoc,
- config da dung,
- output cua run baseline gom `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`,
- link hoac bang chung W&B,
- bao cao ngan phan tich ket qua.

Commit va push cac file bai lam len repository GitHub Classroom:

```bash
git status
git add README.md REPORT_TEMPLATE.md requirements.txt configs docs src ci notebooks
git commit -m "Complete lab 3 UrbanSound8K 1D-CNN"
git push
```
