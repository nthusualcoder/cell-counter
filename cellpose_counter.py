# -*- coding: utf-8 -*-
"""
=============================================================================
Cellpose 기반 자동 세포 카운팅 및 농도 계산
=============================================================================
Hemocytometer 명시야(Brightfield) 화면 캡처에서 Cellpose-SAM(cellpose 4.x, 사전학습 모델)으로 세포를 찾고
위상차로 밝게 빛나는 '생존 세포'만 셉니다 (어둡게 염색된 사멸 세포와 이물질은 개수에서 제외).
 - 경계 규칙: 아래·오른쪽 가장자리에 걸친 세포는 제외합니다 (Hemocytometer 표준 계수법).
 - 격자선 보정: 계수판 격자선 위의 세포는 AI가 잘 놓치므로, 격자선 밖 세포 밀도로 격자선 영역을 보정합니다.

속도 (GTX 1660 Ti 같은 보급형 GPU에서 5초 이내 목표)
 - GPU 세대에 맞는 16비트 연산: Turing(GTX 16xx, RTX 20xx)은 FP16, Ampere 이상은 BF16
 - 캡처의 세포 크기를 먼저 재서(약 0.05초) 세포가 MAX_DIAMETER px 보다 크면 그 크기로 줄여서 분석
 - 프로그램 시작 시 GPU 속도를 재고, 캡처마다 예상 시간이 TIME_BUDGET 초를 넘을 때만 더 줄임
   (라벨링한 사진 기준 세포 수 오차: 원본 크기 2.3 %, 세포 15px 로 축소 시 3.5 %)

[명령줄 사용법]
  python cellpose_counter.py --image <이미지> [--squares 1] [--dilution 2] [--volume 1.0] [--save 결과.png]
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from scipy import ndimage as ndi


def _use_bundled_model_dir():
    """모델 파일(cpsam)이 있는 폴더를 Cellpose 모델 폴더로 지정 (cellpose import 전에 호출).
    updater.model_candidates() 와 같은 순서로 찾습니다."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("CELLPOSE_LOCAL_MODELS_PATH"),
              os.path.join(os.path.dirname(here), "models"),             # CellCounter.exe 설치본
              os.path.join(here, ".cellpose", "models"),                  # 기존 Cellpose_Deploy 폴더
              os.path.join(os.path.expanduser("~"), ".cellpose", "models")):  # Cellpose 기본 위치
        if d and os.path.isfile(os.path.join(d, "cpsam")):
            os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = d
            return


_use_bundled_model_dir()

# ---------------------------------------------------------------------------
# 속도 설정
# ---------------------------------------------------------------------------
MAX_DIAMETER = 22.0     # 세포가 이보다 크면 이 크기로 줄여서 분석 (정확도 손실 없이 빨라짐)
MIN_DIAMETER = 11.0     # 시간이 모자라 줄이더라도 이보다 작게는 줄이지 않음
TIME_BUDGET = 4.5       # 예상 분석 시간 상한(초). 결과 표시까지 5초 안에 끝나도록 여유를 둠
POST_SECONDS = 1.0      # 타일 계산 외 마스크 생성·분류에 드는 예상 시간(초). 실제 분석 시간으로 계속 보정됨
MIN_SIZE = 32           # 이보다 작은 이미지는 분석하지 않음 (px)
TILE = 256              # Cellpose-SAM 타일 크기
TILE_OVERLAP = 0.1
EDGE_PX = 2             # 이 픽셀 이내로 아래·오른쪽 가장자리에 닿으면 경계 규칙 적용

# ---------------------------------------------------------------------------
# 생존/사멸/이물질 분류 상수 (라벨링한 19장 × 화면 배율 4종으로 검증, 크기·밝기 모두 상대값)
# ---------------------------------------------------------------------------
BG_EXCLUDE_DIL = 0.30    # 배경 추정 시 세포를 0.30 × 세포 지름만큼 넓혀서 제외
BG_WINDOW = 6.0          # 국소 배경 = 6 × 세포 지름 창의 평균 밝기
LINE_LEN = 4.0           # 격자선 = 4 × 세포 지름 이상 이어지는 밝은 가로/세로 구조
LINE_REL = 0.12          #          ... 영상 중앙값보다 12 % 이상 밝은 것
CORE_FRAC = 0.55         # 세포 중심부 = 안쪽 거리가 최대값의 55 % 이상인 픽셀
REF_PCT = 75             # 이미지별 기준 밝기 = 세포 크기 물체들의 c_p90 75번째 백분위수
REF_MIN_OBJ = 10         # 기준 물체가 이보다 적으면 REF_DEFAULT 사용
REF_DEFAULT = 0.85
REF_CLIP = (0.45, 1.30)
DARK_P90_FRAC = 0.50     # 어두움 = c_p90 < 0.50 × 기준 (위상차 halo 없음) ...
DARK_CORE_MAX = 0.30     #          ... 이고 중심부가 배경보다 30 % 이상 밝지 않음
DEBRIS_MIN_AREA = 0.40   # 어두운 물체 중 면적 < 0.40 × 중앙값 → 이물질
DEBRIS_MAX_ASPECT = 1.5  # 어두운 물체 중 길쭉함(장축/단축) > 1.5 → 이물질
LINE_BAND = 0.10         # 격자선 보정에 쓰는 띠 = 격자선을 0.10 × 세포 지름만큼 넓힌 영역

LIVE, DEAD, DEBRIS, EDGE = "live", "dead", "debris", "edge"
COLORS = {LIVE: (0, 230, 0), EDGE: (0, 215, 255)}  # BGR. 사멸 세포·이물질은 그리지 않음


def _odd(k):
    k = max(1, int(round(k)))
    return k + 1 - (k % 2)


def _to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def n_tiles(h, w):
    """Cellpose 가 h×w 이미지를 나누는 타일 수. cellpose.core.run_net 은 먼저
    transforms.get_pad_yx(div=16, extra=1, min_size=256) 로 여백을 붙인 뒤 타일로 나눔 (256px → 272px → 2×2)."""
    def n(L):
        padded = 16 * int(np.ceil(L / 16)) + 16 if L >= TILE else max(L + 16, TILE)
        return 1 if padded <= TILE else int(np.ceil((1 + 2 * TILE_OVERLAP) * padded / TILE))
    return n(h) * n(w)


# ---------------------------------------------------------------------------
# 세포 크기 추정 (밝은 구조의 자기상관 폭, 격자선 제외) — 정답 박스 크기 대비 1.00 ± 0.09
# ---------------------------------------------------------------------------
def _coarse_bg(g, win_px):
    H, W = g.shape
    f = max(1, int(win_px // 24))
    sm = cv2.resize(g, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA) if f > 1 else g
    k = min(255, max(3, _odd(win_px / f)))
    med = cv2.medianBlur(np.clip(sm, 0, 255).astype(np.uint8), k).astype(np.float32)
    return cv2.resize(med, (W, H), interpolation=cv2.INTER_LINEAR) if f > 1 else med


def _line_open(img, L):
    L = _odd(L)
    return np.maximum(cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))),
                      cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))))


def estimate_diameter(gray):
    """캡처에서 세포의 대표 지름(px)을 추정"""
    gray = gray.astype(np.float32)
    H, W = gray.shape
    f0 = min(1.0, 1200.0 / max(H, W))
    g = cv2.resize(gray, (max(1, round(W * f0)), max(1, round(H * f0))),
                   interpolation=cv2.INTER_AREA) if f0 < 1 else gray
    h, w = g.shape
    bg = _coarse_bg(g, min(h, w) / 6)
    pos = np.maximum((g - bg) / np.maximum(bg, 1.0), 0)
    pos = np.maximum(pos - _line_open(pos, max(h, w) / 8), 0)
    pos = pos - pos.mean()
    maxr = int(min(64, min(h, w) // 4))
    if maxr < 4:
        return MAX_DIAMETER
    Hp, Wp = cv2.getOptimalDFTSize(h + maxr), cv2.getOptimalDFTSize(w + maxr)
    buf = np.zeros((Hp, Wp), np.float32)
    buf[:h, :w] = pos
    F = cv2.dft(buf, flags=cv2.DFT_COMPLEX_OUTPUT)
    ac = cv2.idft(cv2.mulSpectrums(F, F, 0, conjB=True), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    win = ac[np.ix_(np.r_[Hp - maxr:Hp, 0:maxr + 1], np.r_[Wp - maxr:Wp, 0:maxr + 1])]
    yy, xx = np.mgrid[-maxr:maxr + 1, -maxr:maxr + 1]
    ang = np.degrees(np.arctan2(yy, xx)) % 90
    ok = (ang > 20) & (ang < 70)  # 대각선 방향만 사용 (격자선 잔여물은 가로·세로축에 있음)
    rb = np.round(np.hypot(xx, yy)).astype(int)
    prof = (np.bincount(rb[ok], win[ok], maxr + 2)[:maxr + 1]
            / np.maximum(np.bincount(rb[ok], None, maxr + 2)[:maxr + 1], 1))
    prof[0] = win[maxr, maxr]
    prof = prof / max(prof[0], 1e-12)
    k = int(np.argmax(prof < 0.10))
    rr = float(maxr) if k == 0 else k - 1 + (prof[k - 1] - 0.10) / max(prof[k - 1] - prof[k], 1e-9)
    return float(np.clip(rr / 0.86 / f0, 4.0, 400.0))


# ---------------------------------------------------------------------------
# 생존 세포 판별 (밝은 halo 가 없는 어두운 물체 = 사멸 세포·이물질 → 개수에서 제외)
# ---------------------------------------------------------------------------
def _sum_by_label(lbl, w, n):
    return np.bincount(lbl, weights=w, minlength=n + 1)[1:n + 1]


def _quantile_by_label(lbl, vals, n, q):
    order = np.lexsort((vals, lbl))
    l, v = lbl[order], vals[order]
    ar = np.arange(1, n + 1)
    st = np.searchsorted(l, ar, "left")
    cnt = np.searchsorted(l, ar, "right") - st
    idx = np.clip(st + np.floor(q * np.maximum(cnt - 1, 0)).astype(np.int64), 0, max(len(v) - 1, 0))
    return np.where(cnt > 0, v[idx] if len(v) else 0.0, np.nan)


def object_features(image_bgr, masks):
    """세포(마스크)별 특징값. 밝기는 국소 배경 대비 상대 대비, 크기는 이미지 내 중앙값 대비."""
    masks = np.ascontiguousarray(masks, dtype=np.int32)
    n = int(masks.max())
    if n == 0:
        return {}
    H, W = masks.shape
    g = cv2.cvtColor(_to_bgr(image_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = masks.ravel()
    sel = flat > 0
    lbl = flat[sel]
    area = np.bincount(lbl, minlength=n + 1)[1:].astype(np.float64)
    med_area = float(np.median(area[area > 0]))
    d_med = float(np.sqrt(4.0 * med_area / np.pi))

    # 계수판 격자선: 길게 이어지는 밝은 가로/세로 구조 (가는 선만)
    gmed = float(np.median(g))
    dev = np.clip(g - gmed, 0, None) / max(gmed, 1.0)
    lines = _line_open(dev, max(15, _odd(LINE_LEN * d_med))) > LINE_REL
    # 세포 지름보다 두꺼운 밝은 덩어리(캡처에 들어간 흰 화면 등)는 격자선이 아니라 시료 밖 영역
    k = max(3, _odd(d_med))
    outside = cv2.morphologyEx((dev > LINE_REL).astype(np.uint8), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    outside = cv2.dilate(outside, np.ones((3, 3), np.uint8)) > 0
    lines &= ~outside

    # 국소 배경: 세포(넓힌 영역)·격자선·시료 밖 영역을 뺀 픽셀들의 창 평균
    r = max(1, int(np.ceil(BG_EXCLUDE_DIL * d_med)))
    fgd = cv2.dilate((masks > 0).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ld = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    bgm = ~fgd & ~ld & ~outside
    if bgm.sum() < 0.05 * H * W:
        bgm = ~fgd
    if not bgm.any():
        bgm = np.ones_like(fgd)
    bgf = bgm.astype(np.float32)
    k = _odd(BG_WINDOW * d_med)
    den = cv2.blur(bgf, (k, k))
    num = cv2.blur(g * bgf, (k, k))
    bg = np.where(den > 0.02, num / np.maximum(den, 1e-6), float(np.median(g[bgm])))
    c = ((g - bg) / np.maximum(bg, 1.0)).ravel()[sel].astype(np.float64)  # 배경 대비 상대 밝기

    # 세포 중심부 (붙어 있는 세포끼리는 경계를 분리한 뒤 안쪽 거리 계산)
    dif = np.zeros((H, W), bool)
    dif[:, 1:] |= (masks[:, 1:] != masks[:, :-1]) & (masks[:, :-1] > 0) & (masks[:, 1:] > 0)
    dif[1:, :] |= (masks[1:, :] != masks[:-1, :]) & (masks[:-1, :] > 0) & (masks[1:, :] > 0)
    dt = cv2.distanceTransform(((masks > 0) & ~dif).astype(np.uint8), cv2.DIST_L2, 3).ravel()[sel]
    rin = np.zeros(n)
    np.maximum.at(rin, lbl - 1, dt)
    rin = np.maximum(rin, 1.0)
    core = dt >= CORE_FRAC * rin[lbl - 1]
    ncore = _sum_by_label(lbl[core], np.ones(int(core.sum())), n)
    c_mean = _sum_by_label(lbl, c, n) / np.maximum(area, 1)
    c_core = np.where(ncore > 0, _sum_by_label(lbl[core], c[core], n) / np.maximum(ncore, 1), c_mean)
    c_p90 = np.nan_to_num(_quantile_by_label(lbl, c, n, 0.90), nan=0.0)

    # 무게중심과 길쭉한 정도 (2차 모멘트)
    yy, xx = np.divmod(np.flatnonzero(sel), W)
    xs, ys = xx.astype(np.float64), yy.astype(np.float64)
    a1 = np.maximum(area, 1)
    cx = _sum_by_label(lbl, xs, n) / a1
    cy = _sum_by_label(lbl, ys, n) / a1
    dx, dy = xs - cx[lbl - 1], ys - cy[lbl - 1]
    mxx = _sum_by_label(lbl, dx * dx, n) / a1 + 1 / 12
    myy = _sum_by_label(lbl, dy * dy, n) / a1 + 1 / 12
    mxy = _sum_by_label(lbl, dx * dy, n) / a1
    tr = mxx + myy
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - (mxx * myy - mxy ** 2), 0))
    aspect = np.sqrt((tr / 2 + disc) / np.maximum(tr / 2 - disc, 1e-6))
    return dict(area=area, rel_area=area / med_area, d_med=d_med, c_core=c_core, c_p90=c_p90,
                aspect=aspect, cx=cx, cy=cy, line_mask=lines, outside_mask=outside)


def classify_objects(image_bgr, masks):
    """마스크 번호 k+1 의 분류를 cats[k] 에 'live' / 'dead' / 'debris' 로 반환. (cats, features)"""
    f = object_features(image_bgr, masks)
    if not f:
        return np.zeros(0, "<U6"), f
    ok = (f["area"] > 0) & (f["rel_area"] >= DEBRIS_MIN_AREA)
    ref = (float(np.clip(np.percentile(f["c_p90"][ok], REF_PCT), *REF_CLIP))
           if ok.sum() >= REF_MIN_OBJ else REF_DEFAULT)
    dark = (f["c_p90"] < DARK_P90_FRAC * ref) & (f["c_core"] < DARK_CORE_MAX)
    debris = (dark & ((f["rel_area"] < DEBRIS_MIN_AREA) | (f["aspect"] > DEBRIS_MAX_ASPECT))) | (f["area"] == 0)
    return np.where(debris, DEBRIS, np.where(dark, DEAD, LIVE)), f


class DetectionResult:
    """분석한 크기(원본 × scale)의 Cellpose 마스크 + 세포별 분류.
    경계 규칙 / 격자선 보정은 모델을 다시 돌리지 않고 counts(), status() 에서 바로 계산합니다."""

    def __init__(self, masks, cats, features, scale=1.0, seconds=0.0):
        self.masks = masks
        self.cats = cats
        self.f = features
        self.scale = scale        # 분석 이미지 크기 / 원본 캡처 크기
        self.seconds = seconds    # 분석에 걸린 시간
        on_edge = np.zeros(len(cats), bool)
        if len(cats):
            ids = np.unique(np.concatenate([masks[-EDGE_PX:, :].ravel(), masks[:, -EDGE_PX:].ravel()]))
            on_edge[ids[ids > 0] - 1] = True
        self.on_edge = on_edge

    def status(self, exclude_edge=True):
        """세포별 최종 상태: 'live' / 'dead' / 'debris' / 'edge'(경계 규칙으로 제외된 생존 세포)"""
        st = self.cats.copy()
        if exclude_edge and len(st):
            st[self.on_edge & (st == LIVE)] = EDGE
        return st

    def counts(self, exclude_edge=True, line_correction=True):
        st = self.status(exclude_edge)
        live = st == LIVE
        n_live = int(live.sum())
        estimate = float(n_live)
        if line_correction and n_live:
            # 격자선 띠 밖의 생존 세포 밀도로 띠 영역을 채움: N = (띠 밖 세포 수) / (1 − 띠 면적 비율)
            # (면적 비율은 시료 영역 안에서만 계산 — 캡처에 섞인 흰 화면 등은 제외)
            r = max(1, int(round(LINE_BAND * self.f["d_med"])))
            band = cv2.dilate(self.f["line_mask"].astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            inside = ~self.f["outside_mask"]
            band &= inside
            frac = float(band.sum()) / max(1, int(inside.sum()))
            if 0.0 < frac <= 0.5:
                H, W = band.shape
                px = np.clip(np.round(self.f["cx"]).astype(int), 0, W - 1)
                py = np.clip(np.round(self.f["cy"]).astype(int), 0, H - 1)
                estimate = int((live & ~band[py, px]).sum()) / (1.0 - frac)
        return dict(live=int(round(estimate)), live_detected=n_live, edge=int((st == EDGE).sum()))


# ============================================================================
# CellCounter: Cellpose 모델 로딩 · 세포 탐지 · 농도 계산 · 시각화
# ============================================================================
class CellCounter:
    def __init__(self, use_gpu=True):
        import torch
        from cellpose.models import CellposeModel

        self.torch = torch
        self.gpu, self.device_name, self.precision = False, "CPU", "FP32"
        self.time_factor = 1.0  # 실제 분석 시간 / 예측 시간 (분석할 때마다 갱신)
        if use_gpu and torch.cuda.is_available():
            try:
                self.model = CellposeModel(gpu=True, use_bfloat16=False)  # FP32로 읽은 뒤 GPU 세대에 맞게 변환
                if not self.model.gpu:
                    raise RuntimeError("GPU를 사용할 수 없습니다.")
                self.gpu, self.device_name = True, torch.cuda.get_device_name(0)
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 8:    # Ampere 이상 (RTX 30xx~): BF16
                    self._set_dtype(torch.bfloat16, "BF16")
                elif major >= 7:  # Volta/Turing (GTX 16xx, RTX 20xx): FP16 이 FP32 보다 약 2배 빠름
                    self._set_dtype(torch.float16, "FP16")
                self.overhead, self.sec_per_tile = self._calibrate()
            except Exception:     # 메모리 부족·지원하지 않는 GPU 등 → CPU 로 전환
                self.model = None
                torch.cuda.empty_cache()
                self.gpu, self.device_name, self.precision = False, "CPU", "FP32"
        if not self.gpu:
            self.model = CellposeModel(gpu=False, use_bfloat16=False)
            self.overhead, self.sec_per_tile = self._calibrate()

    def _set_dtype(self, dtype, name):
        self.model.net.dtype = dtype
        self.precision = name

    def _calibrate(self):
        """워밍업 후 분석 시간 = 고정 시간 + 타일 수 × 타일당 시간 의 두 값을 잼.
        16비트 연산 결과가 비정상(NaN)이면 FP32 로 되돌림."""
        def timed(size):
            x = np.zeros((size, size, 3), np.uint8)
            cv2.circle(x, (size // 2, size // 2), 12, (200, 200, 200), 3)
            t = time.perf_counter()
            out = self.model.eval(x)
            return time.perf_counter() - t, out

        small, large = TILE - 16, 2 * TILE - 16  # 여백을 붙이면 256 → 타일 1장, 512 → 타일 9장
        _, out = timed(small)  # 워밍업
        if self.precision != "FP32" and not np.isfinite(np.asarray(out[1][1], np.float32)).all():
            self._set_dtype(self.torch.float32, "FP32")
            timed(small)
        t1, _ = timed(small)
        if not self.gpu:  # CPU 는 느려서 1장만 잼
            return 0.5, t1
        t9, _ = timed(large)
        n1, n9 = n_tiles(small, small), n_tiles(large, large)
        per_tile = max(1e-3, (t9 - t1) / (n9 - n1))
        return max(0.0, t1 - n1 * per_tile), per_tile

    def expected_seconds(self, h, w):
        return (self.overhead + POST_SECONDS + n_tiles(h, w) * self.sec_per_tile) * self.time_factor

    def _analysis_scale(self, h, w, diameter):
        """원본 크기로 분석하되 (세포가 MAX_DIAMETER px 보다 크면 그 크기로 축소),
        예상 시간이 TIME_BUDGET 을 넘을 때만 세포가 MIN_DIAMETER px 가 될 때까지 더 줄임"""
        f = min(1.0, MAX_DIAMETER / diameter)
        f_min = min(f, MIN_DIAMETER / diameter)
        while f > f_min and self.expected_seconds(round(h * f), round(w * f)) > TIME_BUDGET:
            f = max(f_min, f * 0.9)
        return f

    def detect_cells(self, image):
        """BGR(또는 회색조) 캡처에서 세포를 찾아 분류한 DetectionResult 를 반환"""
        t0 = time.perf_counter()
        bgr = _to_bgr(image)
        h, w = bgr.shape[:2]
        if min(h, w) < MIN_SIZE:
            raise ValueError(f"이미지가 너무 작습니다 ({w}×{h}). {MIN_SIZE}×{MIN_SIZE} 픽셀 이상으로 선택해 주세요.")
        scale = self._analysis_scale(h, w, estimate_diameter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        x = bgr if scale >= 1.0 else cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                                                 interpolation=cv2.INTER_AREA)
        predicted = self.expected_seconds(*x.shape[:2])
        masks = self.model.eval(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))[0]
        masks = np.ascontiguousarray(masks, dtype=np.int32)
        cats, features = classify_objects(x, masks)
        seconds = time.perf_counter() - t0
        # 예측이 실제와 다르면(마스크 생성 시간, 다른 프로그램의 GPU 사용 등) 다음 예측에 반영
        ratio = seconds / max(1e-3, predicted / self.time_factor)
        self.time_factor = float(np.clip(0.5 * self.time_factor + 0.5 * ratio, 0.5, 4.0))
        return DetectionResult(masks, cats, features, scale=x.shape[1] / w, seconds=seconds)

    @staticmethod
    def calculate_concentration(count, num_squares=1, dilution_factor=2.0):
        """Hemocytometer: 농도(cells/mL) = 세포 수 ÷ 대구획 수 × 10^4 × 희석 배수"""
        if num_squares <= 0:
            raise ValueError("카운팅한 대구획 수는 0보다 커야 합니다.")
        return count / num_squares * 1e4 * dilution_factor

    @staticmethod
    def visualize(image, detection, exclude_edge=True):
        """센 세포(초록)와 경계 규칙으로 뺀 세포(노랑)의 윤곽선을 원본 크기 캡처 위에 그림"""
        out = _to_bgr(image).copy()
        status = detection.status(exclude_edge)
        inv = 1.0 / detection.scale
        thickness = max(1, round(max(out.shape[:2]) / 1200))
        for i, sl in enumerate(ndi.find_objects(detection.masks)):
            if sl is None or status[i] not in COLORS:
                continue
            obj = (detection.masks[sl] == i + 1).astype(np.uint8)
            contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                           offset=(sl[1].start, sl[0].start))
            contours = [np.round((c + 0.5) * inv - 0.5).astype(np.int32) for c in contours]
            cv2.drawContours(out, contours, -1, COLORS[status[i]], thickness, cv2.LINE_AA)
        return out


# ============================================================================
# 이미지 입출력 (한글 경로 대응)
# ============================================================================
def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    return image


def save_image(path, image):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buf.tofile(path)


# ============================================================================
# 명령줄 실행
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Cellpose 기반 Hemocytometer 생존 세포 카운팅")
    p.add_argument("--image", required=True, help="분석할 이미지 경로")
    p.add_argument("--squares", type=float, default=1, help="이미지에 담긴 대구획(1 mm × 1 mm) 수 [기본 1]")
    p.add_argument("--dilution", type=float, default=2.0, help="희석 배수 (Trypan blue 1:1 → 2) [기본 2]")
    p.add_argument("--volume", type=float, default=1.0, help="현탁액 부피 mL (pellet을 푼 media 양) [기본 1]")
    p.add_argument("--no-edge-rule", action="store_true", help="아래·오른쪽 가장자리 세포도 셈")
    p.add_argument("--no-line-correction", action="store_true", help="격자선 보정 끄기")
    p.add_argument("--save", help="결과 이미지 저장 경로 (.png / .jpg)")
    p.add_argument("--cpu", action="store_true", help="GPU를 쓰지 않음")
    a = p.parse_args()
    if a.squares <= 0 or a.dilution <= 0 or a.volume <= 0:
        sys.exit("대구획 수, 희석 배수, 현탁액 부피는 0보다 커야 합니다.")

    image = load_image(a.image)
    counter = CellCounter(use_gpu=not a.cpu)
    det = counter.detect_cells(image)
    c = det.counts(exclude_edge=not a.no_edge_rule, line_correction=not a.no_line_correction)
    conc = CellCounter.calculate_concentration(c["live"], a.squares, a.dilution)

    print(f"이미지      : {os.path.basename(a.image)} ({image.shape[1]}x{image.shape[0]}, "
          f"{counter.device_name} {counter.precision}, 분석 {det.seconds:.1f}초)")
    print(f"생존 세포 수: {c['live']} (탐지 {c['live_detected']}, 경계 제외 {c['edge']})")
    print(f"세포 농도   : {conc:,.0f} cells/mL (대구획 {a.squares:g}, 희석 ×{a.dilution:g})")
    print(f"총 세포 수  : {conc * a.volume:,.0f} cells (현탁액 {a.volume:g} mL)")
    if a.save:
        save_image(a.save, CellCounter.visualize(image, det, exclude_edge=not a.no_edge_rule))
        print(f"결과 이미지 : {a.save}")


if __name__ == "__main__":
    main()
