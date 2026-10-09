# -*- coding: utf-8 -*-
"""
=============================================================================
Cellpose 기반 자동 세포 카운팅 및 배양액(Media) 계산 프로그램
=============================================================================
Hemocytometer 명시야(Brightfield) 이미지에서 Cellpose를 이용해
세포를 자동으로 탐지·카운팅하고, 세포 농도 및 Media 추가량을 계산합니다.

[필수 설치 명령어 (venv_cellpose 가상환경 내)]
  venv_cellpose\Scripts\python.exe -m pip install cellpose opencv-python numpy matplotlib

[사용법]
  venv_cellpose\Scripts\python.exe cellpose_counter.py --image <이미지경로> [옵션]

[Cellpose 버전 호환]
  - cellpose 4.x (CellposeModel API)
  - cellpose 2.x~3.x (Cellpose API) — 자동 폴백
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 동작하도록 백엔드 설정
import matplotlib.pyplot as plt
from cellpose.utils import outlines_list


# ============================================================================
# Cellpose 버전 호환 모델 로더
# ============================================================================
def _load_cellpose_model(model_type: str, gpu: bool):
    """cellpose 4.x와 2.x~3.x 모두 호환되는 모델 로더."""
    try:
        # cellpose 4.x: CellposeModel (model_type은 무시됨, 기본 모델 사용)
        from cellpose.models import CellposeModel
        return CellposeModel(gpu=gpu)
    except ImportError:
        # cellpose 2.x~3.x: Cellpose
        from cellpose.models import Cellpose
        return Cellpose(model_type=model_type, gpu=gpu)


# ============================================================================
# CellCounter 클래스: Cellpose 기반 세포 탐지·카운팅·농도·희석 계산
# ============================================================================
class CellCounter:
    """
    Cellpose 모델을 활용한 세포 카운팅 및 배양액 계산 클래스.

    Attributes:
        model: Cellpose 추론 모델 인스턴스
        model_type: 사용 중인 모델 종류 (cyto, cyto2 등)
    """

    def __init__(self, model_type: str = "cyto2", use_gpu: bool = True):
        """
        CellCounter 초기화.

        Args:
            model_type: Cellpose 모델 종류 ('cyto', 'cyto2', 'cyto3' 등)
            use_gpu: GPU 사용 여부 (True 권장, 없으면 자동 CPU 폴백)
        """
        self.model_type = model_type
        print(f"[Init] Cellpose 모델 로딩 중... (model={model_type}, gpu={use_gpu})")
        self.model = _load_cellpose_model(model_type, gpu=use_gpu)
        print("[Init] 모델 로딩 완료!")

    # -----------------------------------------------------------------------
    # 1. 세포 탐지 및 카운팅
    # -----------------------------------------------------------------------
    def detect_cells(self, image: np.ndarray, diameter: float = 30.0,
                     flow_threshold: float = 0.4,
                     cellprob_threshold: float = 0.0) -> tuple:
        """
        이미지에서 세포를 탐지하고 마스크와 세포 수를 반환합니다.

        Args:
            image: BGR 또는 Grayscale numpy 배열 이미지
            diameter: 예상 세포 지름 (픽셀 단위, 0이면 자동 추정)
            flow_threshold: flow error 임계값 (낮을수록 엄격)
            cellprob_threshold: 세포 확률 임계값 (높을수록 엄격)

        Returns:
            (masks, total_count) 튜플
            - masks: 세포별 고유 ID가 부여된 마스크 (np.ndarray)
            - total_count: 탐지된 총 세포 수 (int)
        """
        # Cellpose는 RGB 입력을 기대하므로 BGR→RGB 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image  # Grayscale은 그대로 사용

        # 세포 지름이 0이면 자동 추정 모드
        diam = diameter if diameter > 0 else None

        # Cellpose 추론 실행
        masks, flows, styles = self.model.eval(
            img_rgb,
            diameter=diam,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0]  # Brightfield: [grayscale, none]
        )

        total_count = masks.max()  # 마스크의 최대 ID = 총 세포 수
        print(f"[Detect] 탐지된 세포 수: {total_count}개 (diameter={diam}px)")
        return masks, int(total_count)

    # -----------------------------------------------------------------------
    # 2. 세포 농도 계산 (Hemocytometer 표준 공식)
    # -----------------------------------------------------------------------
    @staticmethod
    def calculate_concentration(total_count: int,
                                num_squares: int = 4,
                                dilution_factor: float = 1.0) -> float:
        """
        Hemocytometer 표준 공식으로 세포 농도(cells/mL)를 계산합니다.

        공식: Concentration = (Total count / Number of squares) × 10^4 × Dilution Factor

        Args:
            total_count: 탐지된 총 세포 수
            num_squares: 카운팅한 대구획(Square) 수 (보통 4~5)
            dilution_factor: 희석 배수 (예: Trypan Blue 1:2 희석 → 2)

        Returns:
            세포 농도 (cells/mL)

        Raises:
            ValueError: num_squares가 0 이하인 경우
        """
        if num_squares <= 0:
            raise ValueError("카운팅한 Square 수는 1 이상이어야 합니다.")

        concentration = (total_count / num_squares) * 1e4 * dilution_factor
        print(f"[Conc] 세포 농도: {concentration:,.0f} cells/mL")
        print(f"       (총 {total_count}개 / {num_squares} squares × 10^4 × DF {dilution_factor})")
        return concentration

    # -----------------------------------------------------------------------
    # 3. 추가할 Media 부피 계산 (C1V1 = C2V2)
    # -----------------------------------------------------------------------
    @staticmethod
    def calculate_media_volume(current_concentration: float,
                               current_volume: float,
                               target_concentration: float) -> float:
        """
        C1V1 = C2V2 공식으로 목표 농도 달성에 필요한 추가 Media 부피를 계산합니다.

        공식: V_add = (C_current × V_current / C_target) - V_current

        Args:
            current_concentration: 현재 세포 농도 (cells/mL)
            current_volume: 현재 세포 현탁액 부피 (mL)
            target_concentration: 목표 세포 농도 (cells/mL)

        Returns:
            추가해야 할 Media 부피 (mL)

        Raises:
            ValueError: 목표 농도가 현재 농도 이상인 경우 (농축 필요)
        """
        if target_concentration <= 0:
            raise ValueError("목표 농도는 0보다 커야 합니다.")

        if target_concentration >= current_concentration:
            raise ValueError(
                f"목표 농도({target_concentration:,.0f} cells/mL)가 "
                f"현재 농도({current_concentration:,.0f} cells/mL) 이상입니다.\n"
                "   → Media 추가로는 농도를 높일 수 없습니다. 원심분리 등 농축이 필요합니다."
            )

        v_total = (current_concentration * current_volume) / target_concentration
        v_add = v_total - current_volume

        print(f"[Media] 추가할 Media 부피: {v_add:.2f} mL")
        print(f"        (최종 총 부피: {v_total:.2f} mL → 목표 {target_concentration:,.0f} cells/mL)")
        return v_add

    # -----------------------------------------------------------------------
    # 4. 시각화: 세포 윤곽선 오버레이
    # -----------------------------------------------------------------------
    @staticmethod
    def visualize(image: np.ndarray, masks: np.ndarray,
                  total_count: int, save_path: str = None,
                  show: bool = False) -> np.ndarray:
        """
        원본 이미지 위에 Cellpose가 탐지한 세포 윤곽선을 시각화합니다.

        Args:
            image: 원본 BGR 이미지 (np.ndarray)
            masks: Cellpose 세포 마스크 (np.ndarray)
            total_count: 탐지된 세포 수 (제목 표시용)
            save_path: 결과 이미지 저장 경로 (None이면 저장 안 함)
            show: matplotlib 창으로 표시 여부

        Returns:
            윤곽선이 그려진 BGR 이미지 (np.ndarray)
        """
        # 원본 이미지 복사 (원본 훼손 방지)
        overlay = image.copy()

        # Cellpose 유틸로 각 세포의 윤곽선 좌표 추출
        outlines = outlines_list(masks)

        # 각 세포 윤곽선을 녹색(BGR: 0,255,0)으로 그리기
        for outline in outlines:
            coords = outline.astype(np.int32)
            cv2.polylines(overlay, [coords], isClosed=True,
                          color=(0, 255, 0), thickness=1)

        # matplotlib로 시각화
        if save_path or show:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Cellpose Cell Detection  |  Total: {total_count} cells",
                         fontsize=16, fontweight='bold')

            # 원본 이미지
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image", fontsize=13)
            axes[0].axis('off')

            # 윤곽선 오버레이 이미지
            axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Detected Cells: {total_count}", fontsize=13)
            axes[1].axis('off')

            plt.tight_layout()

            if save_path:
                # 한글 경로 대응: fig.savefig 대신 buffer → cv2 저장
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[Viz] 시각화 결과 저장: {save_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

        return overlay


# ============================================================================
# 이미지 로딩 헬퍼 (한글 경로 대응)
# ============================================================================
def load_image(image_path: str) -> np.ndarray:
    """
    한글 경로를 포함한 이미지를 안전하게 로딩합니다.

    Args:
        image_path: 이미지 파일의 절대/상대 경로

    Returns:
        BGR 형식의 numpy 배열

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: 이미지를 읽을 수 없는 경우
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    # 한글 경로 호환: numpy로 바이트 읽기 → cv2로 디코딩
    img_array = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {image_path}")

    print(f"[Load] 이미지 로딩 완료: {os.path.basename(image_path)} "
          f"({image.shape[1]}x{image.shape[0]}px)")
    return image


# ============================================================================
# CLI 실행 및 테스트 코드
# ============================================================================
if __name__ == "__main__":
    # ----- 커맨드라인 인자 파싱 -----
    parser = argparse.ArgumentParser(
        description="Cellpose 기반 자동 세포 카운팅 및 배양액 계산 프로그램"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="분석할 세포 이미지 경로 (.jpg, .png 등)")
    parser.add_argument("--diameter", type=float, default=30.0,
                        help="예상 세포 지름 (픽셀, 0=자동추정) [기본: 30]")
    parser.add_argument("--squares", type=int, default=4,
                        help="카운팅한 Hemocytometer 대구획 수 [기본: 4]")
    parser.add_argument("--dilution", type=float, default=2.0,
                        help="희석 배수 (예: Trypan Blue 1:2 -> 2) [기본: 2]")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="현재 세포 현탁액 부피 (mL) [기본: 1.0]")
    parser.add_argument("--target", type=float, default=None,
                        help="목표 세포 농도 (cells/mL) [기본: None]")
    parser.add_argument("--model", type=str, default="cyto2",
                        help="Cellpose 모델 종류 [기본: cyto2]")
    parser.add_argument("--no-gpu", action="store_true",
                        help="GPU 비활성화 (CPU만 사용)")
    parser.add_argument("--save", type=str, default=None,
                        help="시각화 결과 저장 경로 [기본: 자동 생성]")
    args = parser.parse_args()

    print("=" * 60)
    print(" Cellpose 자동 세포 카운팅 프로그램")
    print("=" * 60)

    # ----- CellCounter 인스턴스 생성 -----
    counter = CellCounter(model_type=args.model, use_gpu=(not args.no_gpu))

    # ----- 이미지 경로 결정 -----
    if args.image and os.path.exists(args.image):
        image_path = args.image
    else:
        # 이미지가 지정되지 않으면 train_data 폴더에서 첫 번째 이미지 사용
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "train_data")
        if os.path.exists(data_dir):
            imgs = sorted([f for f in os.listdir(data_dir)
                           if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))])
            if imgs:
                image_path = os.path.join(data_dir, imgs[0])
                print(f"\n[Info] 이미지 미지정 -> train_data에서 자동 선택: {imgs[0]}")
            else:
                print("[Error] train_data 폴더에 이미지가 없습니다.")
                sys.exit(1)
        else:
            print("[Error] --image 인자를 지정하거나 train_data 폴더를 준비하세요.")
            sys.exit(1)

    # ----- [Step 1] 이미지 로딩 -----
    print("\n" + "-" * 40)
    print("[Step 1] 이미지 로딩")
    print("-" * 40)
    image = load_image(image_path)

    # ----- [Step 2] Cellpose 세포 탐지 -----
    print("\n" + "-" * 40)
    print("[Step 2] Cellpose 세포 탐지")
    print("-" * 40)
    masks, total_count = counter.detect_cells(image, diameter=args.diameter)

    # ----- [Step 3] 세포 농도 계산 -----
    print("\n" + "-" * 40)
    print("[Step 3] 세포 농도 계산 (Hemocytometer)")
    print("-" * 40)
    concentration = CellCounter.calculate_concentration(
        total_count=total_count,
        num_squares=args.squares,
        dilution_factor=args.dilution
    )

    # ----- [Step 4] Media 부피 계산 (선택) -----
    if args.target is not None:
        print("\n" + "-" * 40)
        print("[Step 4] 추가 Media 부피 계산 (C1V1=C2V2)")
        print("-" * 40)
        try:
            v_add = CellCounter.calculate_media_volume(
                current_concentration=concentration,
                current_volume=args.volume,
                target_concentration=args.target
            )
        except ValueError as e:
            print(f"[Warning] {e}")

    # ----- [Step 5] 시각화 -----
    print("\n" + "-" * 40)
    print("[Step 5] 결과 시각화")
    print("-" * 40)

    save_path = args.save
    if save_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(os.path.dirname(image_path),
                                 f"{base}_cellpose_result.png")

    counter.visualize(image, masks, total_count,
                      save_path=save_path, show=False)

    # ----- 최종 요약 -----
    print("\n" + "=" * 60)
    print(" 분석 결과 요약")
    print("=" * 60)
    print(f"  이미지       : {os.path.basename(image_path)}")
    print(f"  모델         : Cellpose {args.model}")
    print(f"  세포 지름    : {args.diameter}px")
    print(f"  탐지 세포 수 : {total_count}개")
    print(f"  세포 농도    : {concentration:,.0f} cells/mL")
    print(f"    (Squares={args.squares}, DF={args.dilution})")
    if args.target is not None and args.target < concentration:
        v_add = CellCounter.calculate_media_volume(
            concentration, args.volume, args.target)
        print(f"  목표 농도    : {args.target:,.0f} cells/mL")
        print(f"  추가 Media   : {v_add:.2f} mL")
    print(f"  시각화 저장  : {save_path}")
    print("=" * 60)
