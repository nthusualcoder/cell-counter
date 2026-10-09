# -*- coding: utf-8 -*-
"""
=============================================================================
Cellpose 기반 세포 카운팅 GUI (화면 캡처 → 자동 카운팅 → 농도/희석 계산)
=============================================================================
현미경 화면(또는 저장된 사진)에서 Hemocytometer 대구획을 드래그해 고르면
cellpose_counter.CellCounter 로 생존 세포를 세고 농도·총 세포 수·희석량을 계산합니다.
"""

import sys
import os
import math
import traceback
import numpy as np
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QFileDialog,
                               QFormLayout, QLineEdit, QGroupBox, QTextEdit,
                               QMessageBox, QSizePolicy, QCheckBox, QScrollArea, QFrame,
                               QRadioButton, QButtonGroup)
from PySide6.QtGui import (QPixmap, QImage, QFont, QPainter, QPen, QColor,
                           QCursor, QGuiApplication)
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect, QRectF, QTimer

# 현재 디렉토리를 경로에 추가 (포터블 실행 시 로컬 모듈 로딩용)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_DIR)

# pythonw.exe(콘솔 없음)로 실행되면 stdout/stderr 가 None → 라이브러리의 print/진행률 표시가 오류를 내지 않도록
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import updater
from cellpose_counter import CellCounter, load_image, MIN_SIZE


def format_sci(val):
    if val == 0 or not math.isfinite(val):
        return str(val)
    base, exp = f"{val:.2e}".split('e')
    return f"{base} × 10^{int(exp)}"


def qimage_to_bgr(qimg):
    """QImage → OpenCV BGR 배열 (임시 파일 없이 변환)"""
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    h, w, bpl = qimg.height(), qimg.width(), qimg.bytesPerLine()
    arr = np.frombuffer(qimg.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 3]
    return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)


def bgr_to_pixmap(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))


class SnippingWidget(QWidget):
    """화면(또는 열어 둔 사진) 위에서 분석할 영역을 드래그로 고르는 전체 화면 오버레이"""
    capture_complete = Signal(object)  # 고른 영역 (BGR 배열)
    too_small = Signal()
    closed = Signal()

    def __init__(self, image_bgr=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)

        # 마우스가 있는 모니터에 띄움 (다중 모니터 대응)
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.image_bgr = image_bgr
        if image_bgr is None:
            self.source = screen.grabWindow(0)  # 화면 전체 (물리 픽셀)
            self.hint = "현미경 화면에서 분석할 대구획을 드래그하세요    ·    취소: Esc"
        else:
            self.source = bgr_to_pixmap(image_bgr)
            self.hint = "사진에서 분석할 대구획을 드래그하세요    ·    사진 전체: Enter    ·    취소: Esc"

        self.start_point = None
        self.end_point = None
        self.is_drawing = False

    def _source_rect(self):
        """원본(화면 캡처 또는 사진)이 위젯 안에 그려지는 위치 (논리 좌표). 사진은 비율을 유지해 화면에 맞춤."""
        if self.image_bgr is None:
            return QRectF(self.rect())
        sw, sh = self.source.width(), self.source.height()
        k = min(self.width() / sw, self.height() / sh)
        return QRectF((self.width() - sw * k) / 2, (self.height() - sh * k) / 2, sw * k, sh * k)

    def _to_source(self, rect):
        """위젯 영역 → 원본 픽셀 영역 (x, y, w, h). 윈도우 배율(125 %/150 %)과 사진 축소 비율을 반영."""
        r = self._source_rect()
        sx, sy = self.source.width() / r.width(), self.source.height() / r.height()
        x0 = min(max(0, round((rect.left() - r.left()) * sx)), self.source.width())
        y0 = min(max(0, round((rect.top() - r.top()) * sy)), self.source.height())
        x1 = min(max(0, round((rect.right() + 1 - r.left()) * sx)), self.source.width())
        y1 = min(max(0, round((rect.bottom() + 1 - r.top()) * sy)), self.source.height())
        return x0, y0, x1 - x0, y1 - y0

    def _finish(self, x, y, w, h):
        if w < MIN_SIZE or h < MIN_SIZE:
            self.too_small.emit()
        elif self.image_bgr is None:
            self.capture_complete.emit(qimage_to_bgr(self.source.copy(QRect(x, y, w, h)).toImage()))
        else:
            self.capture_complete.emit(self.image_bgr[y:y + h, x:x + w].copy())
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.drawPixmap(self._source_rect(), self.source, QRectF(self.source.rect()))

        dim = QColor(0, 0, 0, 100)  # 선택 영역 바깥을 어둡게
        if self.start_point is None or self.end_point is None:
            painter.fillRect(self.rect(), dim)
        else:
            rect = QRect(self.start_point, self.end_point).normalized()
            w, h = self.width(), self.height()
            t, b, l, r = rect.top(), rect.bottom(), rect.left(), rect.right()
            painter.fillRect(0, 0, w, t, dim)
            painter.fillRect(0, b + 1, w, h - b - 1, dim)
            painter.fillRect(0, t, l, b - t + 1, dim)
            painter.fillRect(r + 1, t, w - r - 1, b - t + 1, dim)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

        # 안내 문구 (오버레이에만 그려지고 분석 이미지에는 들어가지 않음)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        box = painter.fontMetrics().boundingRect(self.hint).adjusted(-16, -8, 16, 8)
        box.moveCenter(QPoint(self.width() // 2, 40))
        painter.fillRect(box, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignCenter, self.hint)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.is_drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.end_point = event.position().toPoint()
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 3 and rect.height() > 3:
                self._finish(*self._to_source(rect))
            else:  # 드래그 없이 클릭만 한 경우 → 다시 고르도록 둠
                self.start_point = self.end_point = None
                self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.image_bgr is not None:
            self._finish(0, 0, self.source.width(), self.source.height())

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class InteractiveImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #444;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.original_pixmap = None

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            super().setPixmap(self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_scaled_pixmap()
        super().resizeEvent(event)


class ModelLoader(QThread):
    """AI 모델을 백그라운드에서 로딩 (로딩 중에도 창이 멈추지 않도록)"""
    loaded = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.loaded.emit(CellCounter(use_gpu=True))
        except Exception:
            self.failed.emit(traceback.format_exc())


class UpdateThread(QThread):
    """GitHub 최신 버전 확인(apply=False) 또는 내려받아 적용(apply=True)"""
    checked = Signal(str, str, bool)   # 현재 버전, 최신 버전, 새 버전 있음
    applied = Signal(str, bool)        # 적용된 버전, 라이브러리 변경 여부
    failed = Signal(str)

    def __init__(self, apply=False):
        super().__init__()
        self.apply = apply

    def run(self):
        try:
            if self.apply:
                self.applied.emit(*updater.download_and_apply(progress=lambda msg: None))
            else:
                self.checked.emit(*updater.check_for_update())
        except Exception as e:
            self.failed.emit(str(e))


class AnalysisThread(QThread):
    done = Signal(object)    # DetectionResult
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, cv_img, counter):
        super().__init__()
        self.cv_img = cv_img
        self.counter = counter

    def run(self):
        try:
            self.progress.emit("세포 탐지 중..." if self.counter.gpu
                               else "세포 탐지 중... (GPU가 없어 CPU로 분석하므로 시간이 걸립니다)")
            self.done.emit(self.counter.detect_cells(self.cv_img))
        except ValueError as e:  # 너무 작은 이미지 등 사용자에게 바로 보여줄 오류
            self.failed.emit(f"⚠️ {e}")
        except Exception:
            self.failed.emit(f"에러 발생:\n{traceback.format_exc()}")


class CellposeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cv_img = None
        self.detection = None   # 마지막 탐지 결과 (마스크 + 세포별 분류)
        self.counter = None
        self.model_error = None
        self.snipper = None
        self.worker = None
        self.pending = False    # 분석 중에 새 이미지가 들어와서, 끝나면 다시 분석해야 함
        self.update_worker = None

        self.initUI()

        self.log("💡 AI 모델 로딩 중... (최초 실행 시 수십 초 걸릴 수 있습니다)")
        self.loader = ModelLoader()
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

        self.check_update(silent=True)  # 시작할 때 새 버전이 있는지 조용히 확인

    def on_model_loaded(self, counter):
        self.counter = counter
        if counter.gpu:
            self.log(f"✅ 모델 로딩 완료! ({counter.device_name}, {counter.precision})\n"
                     "현미경 화면을 띄워두고 [📷 화면 캡처로 바로 카운팅]을 누르세요.")
        else:
            self.log("✅ 모델 로딩 완료 (⚠️ GPU를 찾지 못해 CPU로 분석합니다. 분석에 수십 초가 걸릴 수 있습니다.)")
        if self.cv_img is not None and self.detection is None:
            self._run_thread()  # 로딩 중에 받아 둔 이미지가 있으면 바로 분석

    def on_model_failed(self, msg):
        self.model_error = msg
        self.log(f"[Error] AI 모델 로딩 실패 — 프로그램을 다시 시작해 주세요.\n{msg}")

    # -------------------------------------------------------------
    # 업데이트 (GitHub)
    # -------------------------------------------------------------
    def _start_update_worker(self, apply):
        # 이전 스레드가 시그널을 보낸 직후 아직 끝나지 않았을 수 있으므로 끝날 때까지 기다린 뒤 교체
        if self.update_worker is not None:
            self.update_worker.wait()
        self.update_worker = UpdateThread(apply=apply)
        return self.update_worker

    def check_update(self, silent=False):
        if self.update_worker is not None and self.update_worker.isRunning():
            return
        self.btn_update.setEnabled(False)
        worker = self._start_update_worker(apply=False)
        worker.checked.connect(lambda cur, new, newer: self.on_update_checked(cur, new, newer, silent))
        worker.failed.connect(lambda msg: self.on_update_failed(msg, silent))
        worker.start()

    def on_update_checked(self, cur, new, newer, silent):
        self.btn_update.setEnabled(True)
        if not newer:
            if not silent:
                QMessageBox.information(self, "업데이트", f"최신 버전입니다. (v{cur})")
            return
        self.btn_update.setText(f"🔔 새 버전 v{new} 업데이트")
        self.btn_update.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        if silent:
            return
        if updater.is_dev_checkout():
            QMessageBox.information(self, "업데이트", f"새 버전 v{new}이 있습니다 (현재 v{cur}).\n"
                                    "이 폴더는 개발용 git 폴더이므로 'git pull'로 업데이트하세요.")
            return
        answer = QMessageBox.question(self, "업데이트",
                                      f"새 버전 v{new}이 있습니다 (현재 v{cur}).\n"
                                      "지금 업데이트할까요? 프로그램이 자동으로 다시 시작됩니다.")
        if answer != QMessageBox.Yes:
            return
        self.btn_update.setEnabled(False)
        self.btn_update.setText("⏳ 업데이트 내려받는 중...")
        worker = self._start_update_worker(apply=True)
        worker.applied.connect(self.on_update_applied)
        worker.failed.connect(lambda msg: self.on_update_failed(msg, False))
        worker.start()

    def on_update_applied(self, version, requirements_changed):
        msg = f"v{version} 업데이트를 받았습니다. 프로그램을 다시 시작합니다."
        if requirements_changed:
            msg += "\n(새 라이브러리 설치가 필요해 검은 창에서 설치가 진행됩니다. 창을 닫지 마세요.)"
        QMessageBox.information(self, "업데이트", msg)
        updater.restart_via_bootstrap()
        QApplication.quit()

    def on_update_failed(self, msg, silent):
        self.btn_update.setEnabled(True)
        self.btn_update.setText("🔄 업데이트 확인")
        if not silent:
            QMessageBox.warning(self, "업데이트", f"업데이트하지 못했습니다.\n인터넷 연결을 확인해 주세요.\n\n{msg}")

    def closeEvent(self, event):
        # 업데이트 파일을 복사하는 도중에 끄면 프로그램 파일이 섞일 수 있으므로 기다리게 함
        if self.update_worker is not None and self.update_worker.apply and self.update_worker.isRunning():
            QMessageBox.information(self, "업데이트", "업데이트를 적용하는 중입니다. 잠시 후 다시 닫아 주세요.")
            event.ignore()
            return
        super().closeEvent(event)

    # -------------------------------------------------------------
    # 화면 구성
    # -------------------------------------------------------------
    def initUI(self):
        self.setWindowTitle(f"Cellpose 세포 카운팅 프로그램 v{updater.local_version()}")
        avail = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1300, avail.width() - 40), min(900, avail.height() - 60))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # =========================================================
        # 좌측: 이미지 뷰어 및 버튼부
        # =========================================================
        left_layout = QVBoxLayout()

        self.image_label = InteractiveImageLabel("카운팅할 세포를 캡처해 주세요.")
        self.image_label.setMinimumSize(320, 240)

        self.btn_snip = QPushButton("📷 화면 캡처로 바로 카운팅")
        self.btn_snip.setFont(QFont("Arial", 16, QFont.Bold))
        self.btn_snip.setStyleSheet("background-color: #2196F3; color: white; border-radius: 5px;")
        self.btn_snip.setMinimumHeight(70)
        self.btn_snip.clicked.connect(self.start_snipping)

        self.btn_open = QPushButton("📂 이미지 파일 열기 (사진에서 대구획 선택)")
        self.btn_open.setMinimumHeight(36)
        self.btn_open.clicked.connect(self.open_image_file)

        left_layout.addWidget(self.image_label, stretch=1)
        left_layout.addWidget(self.btn_snip)
        left_layout.addWidget(self.btn_open)

        # =========================================================
        # 우측: 설정 및 결과 패널 (작은 화면에서는 스크롤)
        # =========================================================
        right_widget = QWidget()
        rw_layout = QVBoxLayout(right_widget)
        rw_layout.setContentsMargins(0, 0, 6, 0)

        self.btn_update = QPushButton("🔄 업데이트 확인")
        self.btn_update.clicked.connect(lambda: self.check_update(silent=False))

        self.check_edge_rule = QCheckBox("경계 규칙: 아래·오른쪽 가장자리에 걸친 세포 제외")
        self.check_edge_rule.setToolTip(
            "Hemocytometer 표준 계수법: 위·왼쪽 경계선에 걸친 세포는 세고,\n"
            "아래·오른쪽 경계선에 걸친 세포는 세지 않습니다.\n"
            "대구획 테두리에 맞춰 캡처했을 때 사용하세요.")
        self.check_edge_rule.setChecked(True)
        self.check_edge_rule.toggled.connect(self.on_rule_changed)

        self.check_line_corr = QCheckBox("격자선 보정: 격자선에 가려 놓친 세포를 주변 밀도로 보정")
        self.check_line_corr.setToolTip(
            "AI는 계수판 격자선 위의 세포를 일부 놓칩니다.\n"
            "격자선 밖의 생존 세포 밀도로 격자선 영역의 세포 수를 추정해 더합니다.\n"
            "(라벨링한 사진 19장 기준 세포 수 오차 4.0% → 2.4%)")
        self.check_line_corr.setChecked(True)
        self.check_line_corr.toggled.connect(self.on_rule_changed)

        group_mode = QGroupBox("계산 모드 선택")
        group_mode.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout = QVBoxLayout()

        self.radio_mode1 = QRadioButton("모드 1: 특정 세포 수 채취 (필요한 현탁액 부피 μL 계산)")
        self.radio_mode2 = QRadioButton("모드 2: 새 튜브 제조 (현탁액 분주 후 Media 채움)")
        self.radio_mode1.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_mode1)
        self.mode_group.addButton(self.radio_mode2)

        mode_layout.addWidget(self.radio_mode1)
        mode_layout.addWidget(self.radio_mode2)
        group_mode.setLayout(mode_layout)

        self.radio_mode1.toggled.connect(self.on_mode_changed)

        group_settings = QGroupBox("실시간 부피/농도 설정")
        group_settings.setFont(QFont("Arial", 10, QFont.Bold))
        form_layout = QFormLayout()

        self.input_detected_count = QLineEdit("")
        self.input_detected_count.setFont(QFont("Arial", 12, QFont.Bold))
        self.input_detected_count.setStyleSheet("color: blue;")
        self.input_squares = QLineEdit("1")
        self.input_squares.setToolTip("캡처한 영역에 들어 있는 대구획(1 mm × 1 mm) 개수")
        self.input_dilution = QLineEdit("2")
        self.input_dilution.setToolTip("Trypan blue와 1:1로 섞었으면 2, 섞지 않았으면 1")
        self.input_stock_volume = QLineEdit("1.0")
        self.input_stock_volume.setToolTip("Cell pellet을 풀어준 media의 양 (= 현재 튜브의 세포 현탁액 부피)")
        self.input_target_volume = QLineEdit("4.0")

        target_layout = QHBoxLayout()
        self.input_target_base = QLineEdit("5.0")
        self.input_target_exp = QLineEdit("5")
        self.input_target_base.setAlignment(Qt.AlignCenter)
        self.input_target_exp.setAlignment(Qt.AlignCenter)

        target_layout.addWidget(self.input_target_base)
        target_layout.addWidget(QLabel(" × 10 ^ "))
        target_layout.addWidget(self.input_target_exp)

        for w in (self.input_detected_count, self.input_squares, self.input_dilution,
                  self.input_stock_volume, self.input_target_volume,
                  self.input_target_base, self.input_target_exp):
            w.textChanged.connect(self.update_results_realtime)

        self.label_target_type = QLabel("목표 세포 수 (cells):")
        self.label_target_volume = QLabel("만들 최종 부피 (mL):")

        form_layout.addRow("탐지된 세포 수 (수정 가능):", self.input_detected_count)
        form_layout.addRow("현탁액 부피 (pellet + media, mL):", self.input_stock_volume)
        form_layout.addRow(self.label_target_volume, self.input_target_volume)
        form_layout.addRow(self.label_target_type, target_layout)

        # 초기 상태 설정 (모드 1이 기본이므로 최종 부피 숨김)
        self.label_target_volume.hide()
        self.input_target_volume.hide()

        # 추가기능: 대구획 수(기본 1)·희석 배수(기본 2)는 거의 바꾸지 않으므로 버튼을 눌렀을 때만 보임
        self.btn_advanced = QPushButton()
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.toggled.connect(self.on_advanced_toggled)
        self.advanced_box = QWidget()
        adv_layout = QFormLayout(self.advanced_box)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.addRow("카운팅한 대구획 수:", self.input_squares)
        adv_layout.addRow("희석 배수 (Trypan blue):", self.input_dilution)
        self.advanced_box.hide()
        for w in (self.input_squares, self.input_dilution):
            w.textChanged.connect(self._update_advanced_button)
        self._update_advanced_button()

        settings_layout = QVBoxLayout()
        settings_layout.addLayout(form_layout)
        settings_layout.addWidget(self.btn_advanced)
        settings_layout.addWidget(self.advanced_box)
        group_settings.setLayout(settings_layout)

        info_label = QLabel(
            "<b>[사용 안내]</b><br>"
            "1. <b>[📷 화면 캡처로 바로 카운팅]</b>을 누릅니다.<br>"
            "2. 화면이 어두워지면 현미경 창에서 <b>대구획 테두리에 맞춰</b> 드래그합니다.<br>"
            "3. 마우스를 놓으면 자동으로 카운팅과 계산이 완료됩니다.<br>"
            "<span style='color:#2e7d32'>초록</span>: 센 생존 세포, "
            "<span style='color:#f9a825'>노랑</span>: 경계 규칙으로 제외한 세포<br>"
            "(어둡게 염색된 사멸 세포와 이물질은 세지 않습니다)"
        )
        info_label.setStyleSheet("color: #555; font-size: 13px;")
        info_label.setWordWrap(True)

        self.btn_run = QPushButton("▶ 현재 이미지 다시 분석")
        self.btn_run.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.btn_run.setMinimumHeight(46)
        self.btn_run.clicked.connect(self.run_analysis)

        group_results = QGroupBox("실시간 계산 결과")
        group_results.setFont(QFont("Arial", 10, QFont.Bold))
        res_layout = QVBoxLayout()
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 11))
        self.text_result.setStyleSheet("background-color: #f4f4f4; border: 1px solid #ccc;")
        self.text_result.setMinimumHeight(220)
        res_layout.addWidget(self.text_result)
        group_results.setLayout(res_layout)

        rw_layout.addWidget(self.btn_update)
        rw_layout.addWidget(self.check_edge_rule)
        rw_layout.addWidget(self.check_line_corr)
        rw_layout.addWidget(group_mode)
        rw_layout.addWidget(group_settings)
        rw_layout.addWidget(info_label)
        rw_layout.addWidget(self.btn_run)
        rw_layout.addWidget(group_results, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFixedWidth(490)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(right_scroll)

    def on_mode_changed(self):
        if self.radio_mode1.isChecked():
            self.label_target_type.setText("목표 세포 수 (cells):")
            self.label_target_volume.hide()
            self.input_target_volume.hide()
        else:
            self.label_target_type.setText("목표 농도 (cells/mL):")
            self.label_target_volume.show()
            self.input_target_volume.show()
        self.update_results_realtime()

    def on_advanced_toggled(self, checked):
        self.advanced_box.setVisible(checked)
        self._update_advanced_button()

    def _update_advanced_button(self):
        """버튼 문구 갱신. 접혀 있어도 기본값(대구획 1, 희석 ×2)과 다르면 알 수 있게 표시"""
        try:
            is_default = float(self.input_squares.text()) == 1 and float(self.input_dilution.text()) == 2
        except ValueError:
            is_default = False
        text = "⚙ 추가기능 (대구획 수 · 희석 배수) " + ("▴" if self.btn_advanced.isChecked() else "▾")
        self.btn_advanced.setText(text if is_default else text + "   ⚠ 기본값과 다름")

    def log(self, text, clear=False):
        if clear:
            self.text_result.clear()
        self.text_result.append(text)
        self.text_result.verticalScrollBar().setValue(self.text_result.verticalScrollBar().maximum())

    # -------------------------------------------------------------
    # 이미지 입력 (화면 캡처 / 사진 파일)
    # -------------------------------------------------------------
    def start_snipping(self):
        """메인 윈도우를 숨기고 화면 캡처 오버레이를 엽니다."""
        self.hide()
        QTimer.singleShot(250, self._show_snipper)

    def _show_snipper(self, image_bgr=None):
        self.snipper = SnippingWidget(image_bgr)
        self.snipper.capture_complete.connect(self.set_new_image)
        self.snipper.too_small.connect(self.on_selection_too_small)
        self.snipper.closed.connect(self.show)
        self.snipper.showFullScreen()

    def open_image_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 열기", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "경고", f"이미지를 열 수 없습니다:\n{e}")
            return
        # 저장된 사진은 보통 대구획보다 넓으므로 분석할 대구획을 사진 위에서 고르게 함
        self._show_snipper(image)

    def on_selection_too_small(self):
        self.show()
        self.log(f"⚠️ 선택한 영역이 너무 작습니다 (최소 {MIN_SIZE}×{MIN_SIZE} 픽셀). 다시 선택해 주세요.", clear=True)

    def set_new_image(self, cv_img):
        """새 이미지 수신 후 자동 카운팅 실행"""
        self.show()
        if cv_img is None:
            return
        self.cv_img = cv_img
        self.detection = None
        self.input_detected_count.blockSignals(True)  # 이전 이미지의 세포 수가 남지 않도록 비움
        self.input_detected_count.setText("")
        self.input_detected_count.blockSignals(False)
        self.display_image(cv_img)
        self.log("=" * 40, clear=True)
        self.log(f"✅ 이미지 준비 완료 ({cv_img.shape[1]}×{cv_img.shape[0]}). 자동으로 분석을 시작합니다...")
        if self.model_error:
            self.log(f"[Error] AI 모델 로딩에 실패해 분석할 수 없습니다. 프로그램을 다시 시작해 주세요.\n{self.model_error}")
            return
        if self.counter is None:
            self.log("⏳ AI 모델이 아직 로딩 중입니다. 로딩이 끝나면 자동으로 분석합니다.")
            return
        self._run_thread()

    def display_image(self, cv_img):
        self.image_label.set_image(bgr_to_pixmap(cv_img))

    # -------------------------------------------------------------
    # 분석 및 연산 로직
    # -------------------------------------------------------------
    def run_analysis(self):
        """수동으로 [다시 분석] 버튼을 눌렀을 때의 동작"""
        if self.cv_img is None:
            QMessageBox.warning(self, "경고", "먼저 화면을 캡처하거나 이미지 파일을 열어 주세요!")
            return
        if self.model_error:
            QMessageBox.warning(self, "경고", "AI 모델 로딩에 실패했습니다. 프로그램을 다시 시작해 주세요.")
            return
        if self.counter is None:
            QMessageBox.warning(self, "경고", "AI 모델이 아직 로딩 중입니다. 잠시 후 다시 시도해 주세요.")
            return
        self.log("=" * 40, clear=True)
        self.log("현재 이미지를 다시 분석합니다...")
        self._run_thread()

    def _run_thread(self):
        self.btn_run.setEnabled(False)
        self.btn_snip.setEnabled(False)
        self.btn_open.setEnabled(False)
        if self.worker is not None and self.worker.isRunning():
            self.pending = True  # 지금 분석이 끝나면 그 결과는 버리고 최신 이미지를 분석
            return
        self.pending = False
        worker = AnalysisThread(self.cv_img, self.counter)
        worker.progress.connect(lambda msg, w=worker: w is self.worker and not self.pending and self.log(msg))
        worker.done.connect(lambda det, w=worker: self.handle_finished(w, det))
        worker.failed.connect(lambda msg, w=worker: self.handle_error(w, msg))
        worker.finished.connect(self._on_worker_finished)
        self.worker = worker
        worker.start()

    def _on_worker_finished(self):
        if self.pending:
            self._run_thread()

    def handle_error(self, worker, err_msg):
        if worker is not self.worker or self.pending:
            return
        self.log(err_msg)
        self.finish_ui_reset()

    def handle_finished(self, worker, detection):
        if worker is not self.worker or self.pending:
            return  # 분석하는 사이에 새 이미지가 들어옴 → 이 결과는 버림
        self.detection = detection
        self.finish_ui_reset()
        self.on_rule_changed()

    def _counts(self):
        return self.detection.counts(exclude_edge=self.check_edge_rule.isChecked(),
                                     line_correction=self.check_line_corr.isChecked())

    def on_rule_changed(self):
        """경계 규칙 / 격자선 보정 변경 → 모델 재실행 없이 개수와 그림만 다시 계산"""
        if self.detection is None or self.cv_img is None:
            return
        self.display_image(CellCounter.visualize(self.cv_img, self.detection,
                                                 exclude_edge=self.check_edge_rule.isChecked()))
        # 값이 같으면 textChanged 가 오지 않으므로 신호를 막고 넣은 뒤 직접 갱신
        self.input_detected_count.blockSignals(True)
        self.input_detected_count.setText(str(self._counts()["live"]))
        self.input_detected_count.blockSignals(False)
        self.update_results_realtime()

    def _detection_summary(self):
        if self.detection is None:
            return []
        c = self._counts()
        lines = []
        if c["live"] != c["live_detected"]:
            lines.append(f"   화면의 초록 세포 {c['live_detected']}개 + 격자선 보정 {c['live'] - c['live_detected']:+d}개")
        if c["edge"]:
            lines.append(f"   경계 규칙으로 {c['edge']}개 제외 (아래·오른쪽 가장자리)")
        lines.append(f"   분석 시간 {self.detection.seconds:.1f}초")
        return lines

    def update_results_realtime(self):
        if not self.input_detected_count.text().strip():
            self.log("⏳ 아직 세포 수가 없습니다 (분석 중이거나 분석하지 못했습니다).\n"
                     "   세포 수를 직접 입력해도 계산됩니다.", clear=True)
            return
        try:
            count = int(self.input_detected_count.text())
            squares = float(self.input_squares.text())
            dilution = float(self.input_dilution.text())
            v_stock = float(self.input_stock_volume.text())
            target = float(self.input_target_base.text()) * 10 ** float(self.input_target_exp.text())
            v_target = float(self.input_target_volume.text()) if self.radio_mode2.isChecked() else 1.0
        except (ValueError, OverflowError):
            self.log("⚠️ 입력값을 확인해 주세요 (숫자만 입력).", clear=True)
            return
        if not all(math.isfinite(v) for v in (squares, dilution, v_stock, target, v_target)):
            self.log("⚠️ 입력값이 너무 큽니다.", clear=True)
            return
        if count < 0 or squares <= 0 or dilution <= 0 or v_stock <= 0:
            self.log("⚠️ 세포 수는 0 이상, 대구획 수·희석 배수·현탁액 부피는 0보다 커야 합니다.", clear=True)
            return

        # 혈구계산판 농도 = 현탁액 자체의 농도 (cells/mL) → 현탁액 부피와 무관
        # 총 세포 수 = 농도 × 현탁액 부피 → pellet을 몇 mL로 풀었는지에 따라 달라짐
        concentration = CellCounter.calculate_concentration(count, squares, dilution)
        total_cells = concentration * v_stock

        lines = ["=" * 40,
                 f"▶ 세포 수: {count} 개  (대구획 {squares:g}개, 희석 ×{dilution:g})"]
        lines += self._detection_summary()
        lines += ["-" * 40,
                  f"세포 농도 : {format_sci(concentration)} cells/mL",
                  f"총 세포 수: {format_sci(total_cells)} cells (현탁액 {v_stock:g} mL)",
                  "  ※ 농도는 계수판에서 직접 잰 값이라 부피와 무관하고,",
                  "    총 세포 수 = 농도 × 현탁액 부피 입니다.",
                  "-" * 40]

        if count == 0:
            lines.append("세포가 0개라 채취량/희석량을 계산할 수 없습니다.")
        elif target <= 0:
            lines.append("계산 불가 (목표값이 0 이하)")
        elif self.radio_mode1.isChecked():
            v_extract_ml = target / concentration
            lines += [f"목표 세포 수 : {format_sci(target)} cells",
                      f"현탁액 채취량: {v_extract_ml * 1000:,.1f} μL"]
            if v_extract_ml > v_stock:
                lines.append(f"⚠️ 현탁액 전체({v_stock:g} mL)를 써도 부족합니다!")
            else:
                lines.append(f"  (현탁액 {v_stock:g} mL로 {math.floor(v_stock / v_extract_ml + 1e-9)}번 채취 가능)")
        elif v_target <= 0:
            lines.append("계산 불가 (만들 최종 부피가 0 이하)")
        elif target > concentration:
            lines += [f"목표 농도({format_sci(target)})가 현재 농도보다 높아 희석으로는 만들 수 없습니다.",
                      f"→ 원심분리 후 {total_cells / target:.3f} mL로 재현탁하면 목표 농도가 됩니다."]
        else:
            v_extract_ml = target * v_target / concentration
            lines += [f"현탁액 투입: {v_extract_ml * 1000:,.1f} μL",
                      f"추가 Media : {v_target - v_extract_ml:.3f} mL",
                      f"최종: {format_sci(target)} cells/mL 용액 {v_target:g} mL"]
            if v_extract_ml > v_stock:
                lines.append(f"⚠️ 현탁액({v_stock:g} mL)보다 많이 필요합니다! "
                             f"(최대 {total_cells / target:.2f} mL까지 제조 가능)")
        lines.append("=" * 40)
        self.log("\n".join(lines), clear=True)

    def finish_ui_reset(self):
        self.btn_run.setEnabled(True)
        self.btn_snip.setEnabled(True)
        self.btn_open.setEnabled(True)


def _excepthook(exc_type, exc, tb):
    """처리되지 않은 오류를 error.log 에 남기고 창으로 보여줌 (콘솔 없이 실행될 때 대비)"""
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        with open(os.path.join(APP_DIR, "error.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "오류", msg[-3000:])


if __name__ == '__main__':
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)

    font = app.font()
    font.setFamily("Malgun Gothic")
    app.setFont(font)

    ex = CellposeGUI()
    ex.show()
    # 모델 로딩·분석 스레드가 아직 돌고 있어도 안전하게 종료 (파이썬 정리 단계에서 Qt 강제 종료 방지)
    os._exit(app.exec())
