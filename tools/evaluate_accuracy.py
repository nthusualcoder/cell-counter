# -*- coding: utf-8 -*-
"""
라벨링한 사진으로 세포 카운팅 정확도를 확인합니다.

  python tools/evaluate_accuracy.py [--scale 0.25] [--data <폴더>]

<폴더>/train_data/<이름>.jpg 와 <폴더>/labels/<이름>.txt (YOLO 형식 박스, 한 줄 = 생존 세포 1개)가 필요합니다.
화면 캡처 크기를 흉내 내려고 원본 사진을 --scale 배로 줄여서 분석합니다 (0.25 ≈ 세포 지름 18 px).
라벨은 사진 테두리에 걸친 세포도 포함하므로 경계 규칙은 끄고 비교합니다.
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cellpose_counter import CellCounter, load_image, LIVE  # noqa: E402


def matched(xy, boxes, w, h):
    """예측 세포 중심과 라벨 박스 중심을 1:1 매칭 (박스 크기의 0.6배 이내)"""
    if len(xy) == 0 or len(boxes) == 0:
        return 0
    gt = np.stack([boxes[:, 1] * w, boxes[:, 2] * h], 1)
    radius = 0.6 * np.maximum(boxes[:, 3] * w, boxes[:, 4] * h)
    d = np.linalg.norm(xy[:, None, :] - gt[None, :, :], axis=2)
    cost = np.where(d <= radius[None, :], d, 1e6)
    r, c = linear_sum_assignment(cost)
    return int(np.sum(cost[r, c] < 1e6))


def summary(name, pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    err = (pred - gt) / gt
    print(f"  {name:28s} 평균 오차 {100 * np.abs(err).mean():4.1f}%  최대 {100 * np.abs(err).max():4.1f}%  "
          f"치우침 {100 * err.mean():+5.1f}%")


def main():
    p = argparse.ArgumentParser(description="라벨링한 사진으로 정확도 확인")
    p.add_argument("--data", default=ROOT, help="train_data, labels 폴더가 있는 곳 [기본: 저장소 폴더]")
    p.add_argument("--scale", type=float, default=0.25, help="화면 캡처 배율 흉내 [기본 0.25]")
    p.add_argument("--min-labels", type=int, default=20, help="라벨이 이보다 적은 사진은 미완성으로 보고 제외")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()

    items = []
    for lp in sorted(glob.glob(os.path.join(a.data, "labels", "*.txt"))):
        name = os.path.splitext(os.path.basename(lp))[0]
        boxes = np.loadtxt(lp, ndmin=2)
        ip = os.path.join(a.data, "train_data", name + ".jpg")
        if len(boxes) < a.min_labels or not os.path.isfile(ip):
            print(f"제외: {name} (라벨 {len(boxes)}개)")
            continue
        items.append((name, ip, boxes))
    if not items:
        sys.exit("평가할 사진이 없습니다.")

    counter = CellCounter(use_gpu=not a.cpu)
    rows = []
    print(f"{'사진':24s} {'정답':>5s} {'전체':>5s} {'생존':>5s} {'보정':>5s}  {'분석 크기':>11s} {'초':>5s}")
    for name, ip, boxes in items:
        img = cv2.resize(load_image(ip), None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        det = counter.detect_cells(img)
        c = det.counts(exclude_edge=False, line_correction=True)
        live = det.cats == LIVE
        xy = np.stack([det.f["cx"], det.f["cy"]], 1)[live] / det.scale if len(det.cats) else np.zeros((0, 2))
        h, w = img.shape[:2]
        tp = matched(xy, boxes, w, h)
        rows.append((len(boxes), len(det.cats), c["live_detected"], c["live"], tp, det.seconds))
        size = f"{det.masks.shape[1]}x{det.masks.shape[0]}"
        print(f"{name[:24]:24s} {len(boxes):5d} {len(det.cats):5d} {c['live_detected']:5d} {c['live']:5d}  "
              f"{size:>11s} {det.seconds:5.1f}")

    gt, allobj, live, corr, tp, dt = map(np.array, zip(*rows))
    prec, rec = tp.sum() / max(1, live.sum()), tp.sum() / gt.sum()
    print(f"\n[배율 {a.scale}] 사진 {len(rows)}장, 사진당 평균 {dt.mean():.1f}초 "
          f"({counter.device_name} {counter.precision})")
    print(f"  생존 세포 탐지: 정밀도 {prec:.3f}, 재현율 {rec:.3f}, F1 {2 * prec * rec / (prec + rec):.3f}")
    summary("필터 없음 (이전 버전 방식)", allobj, gt)
    summary("생존 세포만", live, gt)
    summary("생존 세포 + 격자선 보정", corr, gt)


if __name__ == "__main__":
    main()
