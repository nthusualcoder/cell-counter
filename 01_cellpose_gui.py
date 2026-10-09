# -*- coding: utf-8 -*-
"""
=============================================================================
Cellpose 기반 세포 카운팅 전용 GUI 애플리케이션 (캡처 도구 전용)
=============================================================================
터미널 명령어 없이 직관적인 UI를 통해 캡처하고 분석합니다.
내부적으로 cellpose_counter.py 의 기능과 GPU(CUDA)를 활용합니다.
"""

import sys
import os
import time
import cv2
import numpy as np
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton,
                               QFormLayout, QLineEdit, QGroupBox, QTextEdit,
                               QMessageBox, QSizePolicy, QCheckBox,
                               QRadioButton, QButtonGroup)
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor, QScreen
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect, QTimer

# 현재 디렉토리를 경로에 추가 (포터블 실행 시 로컬 모듈 로딩용)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cellpose_counter import CellCounter

def format_sci(val):
    if val == 0:
        return "0"
    s = f"{val:.1e}"
    base, exp = s.split('e')
    exp = int(exp)
    return f"{base} * 10^{exp}"


class SnippingWidget(QWidget):
    """화면 캡처를 위한 오버레이 위젯 (Snipping Tool)"""
    capture_complete = Signal(object)  # cv2 이미지 반환
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        
        # 전체 화면 캡처
        screen = QApplication.primaryScreen()
        self.full_screen_pixmap = screen.grabWindow(0)
        self.setGeometry(screen.geometry())
        
        self.start_point = None
        self.end_point = None
        self.is_drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.full_screen_pixmap)
        
        # 반투명 오버레이
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.NoPen)
        
        if not self.start_point or not self.end_point:
            painter.drawRect(self.rect())
            return
            
        rect = QRect(self.start_point, self.end_point).normalized()
        
        w, h = self.width(), self.height()
        t, b, l, r = rect.top(), rect.bottom(), rect.left(), rect.right()
        
        # 어두운 바깥 영역 그리기
        painter.drawRect(0, 0, w, t)
        painter.drawRect(0, b + 1, w, h - b - 1)
        painter.drawRect(0, t, l, b - t + 1)
        painter.drawRect(r + 1, t, w - r - 1, b - t + 1)
        
        # 선택된 영역에 빨간 테두리
        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

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
            self.update()
            
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 10 and rect.height() > 10:
                cropped = self.full_screen_pixmap.copy(rect)
                
                # 안전한 변환을 위해 임시 파일 사용
                tmp_path = os.path.join(tempfile.gettempdir(), 'snip_temp.png')
                cropped.save(tmp_path, 'PNG')
                
                # cv2 이미지로 로드
                cv_img = cv2.imread(tmp_path)
                if cv_img is not None:
                    self.capture_complete.emit(cv_img)
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

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
        self.scaled_pixmap = None

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            self.scaled_pixmap = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(self.scaled_pixmap)

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_scaled_pixmap()
        super().resizeEvent(event)


class AnalysisThread(QThread):
    finished = Signal(object, int)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, cv_img, counter_instance, fast_mode=True):
        super().__init__()
        self.cv_img = cv_img  # 직접 전달된 cv2 이미지
        self.counter = counter_instance
        self.fast_mode = fast_mode

    def run(self):
        try:
            self.progress.emit("1. 이미지 분석 준비 중...")
            image = self.cv_img
            
            self.progress.emit("2. 세포 탐지 중 (GPU 연산)...")
            
            if self.fast_mode:
                orig_crop_h, orig_crop_w = image.shape[:2]
                small_crop = cv2.resize(image, (orig_crop_w // 2, orig_crop_h // 2), interpolation=cv2.INTER_AREA)
                masks, total_count = self.counter.detect_cells(small_crop, diameter=15.0)
                
                self.progress.emit("3. 결과 렌더링 중...")
                small_result = self.counter.visualize(small_crop, masks, total_count, save_path=None, show=False)
                result_crop = cv2.resize(small_result, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_LINEAR)
            else:
                masks, total_count = self.counter.detect_cells(image, diameter=30.0)
                self.progress.emit("3. 결과 렌더링 중...")
                result_crop = self.counter.visualize(image, masks, total_count, save_path=None, show=False)
            
            self.finished.emit(result_crop, int(total_count))
            
        except Exception as e:
            import traceback
            self.error.emit(f"에러 발생:\n{str(e)}\n\n{traceback.format_exc()}")


class CellposeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cv_img = None
        self.cv_image_shape = None
        self.last_total_count = 0
        self.counter = None
        self.snipper = None
        
        self.initUI()
        
        QTimer.singleShot(100, self.preload_model)

    def preload_model(self):
        self.log("💡 최초 1회 AI 모델 로딩 중... (잠시만 기다려주세요)")
        try:
            self.counter = CellCounter(model_type="cyto2", use_gpu=True)
            self.log("✅ 모델 로딩 완료! 즉시 카운팅이 가능합니다.\n현미경 화면을 띄워두고 [📷 화면 캡처로 바로 카운팅]을 누르세요.")
        except Exception as e:
            self.log(f"[Error] 모델 로딩 실패: {e}")

    def initUI(self):
        self.setWindowTitle("Cellpose 고속 카운팅 프로그램 (화면 캡처 전용)")
        self.resize(1300, 850)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # =========================================================
        # 좌측: 이미지 뷰어 및 버튼부
        # =========================================================
        left_layout = QVBoxLayout()
        
        self.image_label = InteractiveImageLabel("카운팅할 세포를 캡처해 주세요.")
        self.image_label.setMinimumSize(600, 600)
        
        self.btn_snip = QPushButton("📷 화면 캡처로 바로 카운팅")
        self.btn_snip.setFont(QFont("Arial", 16, QFont.Bold))
        self.btn_snip.setStyleSheet("background-color: #2196F3; color: white; border-radius: 5px;")
        self.btn_snip.setMinimumHeight(80)
        self.btn_snip.clicked.connect(self.start_snipping)
        
        left_layout.addWidget(self.image_label, stretch=1)
        left_layout.addWidget(self.btn_snip)
        
        # =========================================================
        # 우측: 설정 및 결과 패널
        # =========================================================
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setFixedWidth(450)
        rw_layout = QVBoxLayout(right_widget)
        rw_layout.setContentsMargins(0, 0, 0, 0)
        
        self.check_fast_mode = QCheckBox("고속 모드 (50% 축소로 4배 이상 속도 향상)")
        self.check_fast_mode.setChecked(True)
        self.check_fast_mode.setStyleSheet("color: #d32f2f; font-weight: bold; font-size: 14px;")
        
        group_mode = QGroupBox("계산 모드 선택")
        group_mode.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout = QVBoxLayout()
        
        self.radio_mode1 = QRadioButton("모드 1: 특정 세포 수 채취 (필요한 원액 부피 μL 계산)")
        self.radio_mode2 = QRadioButton("모드 2: 새 튜브 제조 (원액 분주 후 Media 채움)")
        self.radio_mode1.setChecked(True)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_mode1)
        self.mode_group.addButton(self.radio_mode2)
        
        mode_layout.addWidget(self.radio_mode1)
        mode_layout.addWidget(self.radio_mode2)
        group_mode.setLayout(mode_layout)
        
        self.radio_mode1.toggled.connect(self.on_mode_changed)
        self.radio_mode2.toggled.connect(self.on_mode_changed)

        group_settings = QGroupBox("실시간 부피/농도 설정")
        group_settings.setFont(QFont("Arial", 10, QFont.Bold))
        form_layout = QFormLayout()
        
        self.input_stock_volume = QLineEdit("1.0")
        self.input_target_volume = QLineEdit("4.0")
        self.input_detected_count = QLineEdit("0")
        self.input_detected_count.setFont(QFont("Arial", 12, QFont.Bold))
        self.input_detected_count.setStyleSheet("color: blue;")
        
        target_layout = QHBoxLayout()
        self.input_target_base = QLineEdit("5.0")
        self.input_target_exp = QLineEdit("5")
        self.input_target_base.setAlignment(Qt.AlignCenter)
        self.input_target_exp.setAlignment(Qt.AlignCenter)
        
        target_layout.addWidget(self.input_target_base)
        target_layout.addWidget(QLabel(" × 10 ^ "))
        target_layout.addWidget(self.input_target_exp)
        
        self.input_stock_volume.textChanged.connect(self.update_results_realtime)
        self.input_target_volume.textChanged.connect(self.update_results_realtime)
        self.input_target_base.textChanged.connect(self.update_results_realtime)
        self.input_target_exp.textChanged.connect(self.update_results_realtime)
        self.input_detected_count.textChanged.connect(self.update_results_realtime)
        
        self.label_target_type = QLabel("목표 세포 수 (cells):")
        self.label_target_volume = QLabel("만들 최종 부피 (mL):")
        
        form_layout.addRow("탐지된 세포 수 (수정 가능):", self.input_detected_count)
        form_layout.addRow("원액 총 부피 (mL):", self.input_stock_volume)
        form_layout.addRow(self.label_target_volume, self.input_target_volume)
        form_layout.addRow(self.label_target_type, target_layout)
        
        # 초기 상태 설정 (모드 1이 기본이므로 부피 숨김)
        self.label_target_volume.hide()
        self.input_target_volume.hide()
        
        group_settings.setLayout(form_layout)
        
        info_label = QLabel(
            "<b>[사용 안내]</b><br>"
            "1. 좌측 하단의 <b>[📷 화면 캡처로 바로 카운팅]</b>을 누릅니다.<br>"
            "2. 화면이 어두워지면 현미경 창의 세포 영역을 마우스로 드래그합니다.<br>"
            "3. 마우스를 놓는 즉시 <b>자동으로 카운팅과 결과 계산</b>이 완료됩니다."
        )
        info_label.setStyleSheet("color: #555; font-size: 13px;")
        info_label.setWordWrap(True)
        
        self.btn_run = QPushButton("▶ 현재 캡처된 화면 다시 분석")
        self.btn_run.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.btn_run.setMinimumHeight(60)
        self.btn_run.clicked.connect(self.run_analysis)
        
        group_results = QGroupBox("실시간 계산 결과")
        group_results.setFont(QFont("Arial", 10, QFont.Bold))
        res_layout = QVBoxLayout()
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 12))
        self.text_result.setStyleSheet("background-color: #f4f4f4; border: 1px solid #ccc;")
        res_layout.addWidget(self.text_result)
        group_results.setLayout(res_layout)
        
        rw_layout.addWidget(self.check_fast_mode)
        rw_layout.addWidget(group_mode)
        rw_layout.addWidget(group_settings)
        rw_layout.addWidget(info_label)
        rw_layout.addWidget(self.btn_run)
        rw_layout.addWidget(group_results)
        
        right_layout.addWidget(right_widget)
        right_layout.addStretch(1)
        
        main_layout.addLayout(left_layout, stretch=7)
        main_layout.addLayout(right_layout, stretch=3)

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

    # 누락되었던 로그 함수 복원!
    def log(self, text, clear=False):
        if clear:
            self.text_result.clear()
        self.text_result.append(text)
        self.text_result.verticalScrollBar().setValue(self.text_result.verticalScrollBar().maximum())

    # -------------------------------------------------------------
    # 캡처 도구 (Snipping) 관련 로직
    # -------------------------------------------------------------
    def start_snipping(self):
        """메인 윈도우를 숨기고 캡처 위젯을 엽니다."""
        self.hide()
        QTimer.singleShot(200, self._show_snipper)

    def _show_snipper(self):
        self.snipper = SnippingWidget()
        self.snipper.capture_complete.connect(self.on_snip_completed)
        self.snipper.closed.connect(self.show)
        self.snipper.showFullScreen()

    def on_snip_completed(self, cv_img):
        """캡처 완료 후 이미지 수신 및 자동 카운팅 실행"""
        self.show()
        if cv_img is None:
            return
            
        self.cv_img = cv_img
        self.cv_image_shape = cv_img.shape
        self.last_total_count = 0
        self.text_result.clear()
        
        self.log("="*40, clear=True)
        self.log("✅ 화면 캡처 완료! 자동으로 분석을 시작합니다...")
        
        self.display_image(cv_img)
        
        # 캡처된 이미지는 그 자체가 분석 대상이므로 전체를 바로 분석
        self._run_thread()

    def display_image(self, cv_img):
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, c = cv_img_rgb.shape
        bytesPerLine = c * w
        qImg = QImage(cv_img_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.set_image(pixmap)

    # -------------------------------------------------------------
    # 분석 및 연산 로직
    # -------------------------------------------------------------
    def run_analysis(self):
        """수동으로 [다시 분석] 버튼을 눌렀을 때의 동작"""
        if self.cv_img is None:
            QMessageBox.warning(self, "경고", "먼저 [화면 캡처로 바로 카운팅]을 진행해 주세요!")
            return
            
        if self.counter is None:
            QMessageBox.warning(self, "경고", "AI 모델이 아직 로딩 중입니다. 잠시 후 다시 시도해 주세요.")
            return

        self.log("="*40, clear=True)
        self.log("캡처된 이미지를 다시 분석합니다...")
            
        self._run_thread()

    def _run_thread(self):
        self.btn_run.setEnabled(False)
        self.btn_snip.setEnabled(False)
        
        self.thread = AnalysisThread(
            cv_img=self.cv_img,
            counter_instance=self.counter,
            fast_mode=self.check_fast_mode.isChecked()
        )
        self.thread.progress.connect(self.log)
        self.thread.error.connect(self.handle_error)
        self.thread.finished.connect(self.handle_finished)
        self.thread.start()

    def handle_error(self, err_msg):
        self.log(err_msg)
        self.finish_ui_reset()

    def handle_finished(self, result_img, total_count):
        self.last_total_count = total_count
        self.input_detected_count.setText(str(total_count))
        self.display_image(result_img)
        self.update_results_realtime()
        self.finish_ui_reset()

    def update_results_realtime(self):
        try:
            current_count = int(self.input_detected_count.text())
            v_stock = float(self.input_stock_volume.text())
            v_target = float(self.input_target_volume.text())
            base = float(self.input_target_base.text())
            exp = float(self.input_target_exp.text())
            target_conc = base * (10 ** exp)
        except ValueError:
            # 입력 중일 때는 에러 메시지를 띄우지 않고 대기
            return

        concentration = (current_count / 1.0) * 1e4 * 2.0
        
        conc_sci = format_sci(concentration)
        target_sci = format_sci(target_conc)
        
        self.log("="*40, clear=True)
        self.log(f"▶ 최종 세포 수: {current_count} 개")
        self.log("-" * 40)

        if self.radio_mode1.isChecked():
            # 목표 농도 입력값을 '목표 세포 수'로 간주
            target_cell_count = target_conc
            if target_cell_count <= 0:
                self.log("계산 불가 (목표 세포 수 0)")
            else:
                v_extract_ml = target_cell_count / concentration
                v_extract_ul = v_extract_ml * 1000
                warning = ""
                if v_extract_ml > v_stock:
                    warning = f" (⚠️경고: 원액 {v_stock}mL 보다 많이 필요함!)"
                
                self.log(
                    f"현재 농도: {conc_sci}\n"
                    f"목표 세포 수: {format_sci(target_cell_count)} cells\n"
                    f"원액 채취량: {v_extract_ul:,.1f} μL{warning}"
                )
        else:
            if target_conc <= 0:
                self.log("계산 불가 (목표 농도 0)")
            elif target_conc >= concentration:
                self.log(f"현재 농도: {conc_sci}\n결과: 농축 필요 (원액 농도가 부족함)")
            else:
                v_extract_ml = (target_conc * v_target) / concentration
                v_extract_ul = v_extract_ml * 1000
                v_media_add = v_target - v_extract_ml
                
                warning = ""
                if v_extract_ml > v_stock:
                    warning = f" (⚠️경고: 원액 {v_stock}mL 보다 많이 필요함!)"
                    
                self.log(
                    f"현재 농도: {conc_sci}\n"
                    f"원액 투입: {v_extract_ul:,.1f} μL{warning}\n"
                    f"추가 Media: {v_media_add:.3f} mL\n"
                    f"최종: {target_sci} cells/mL 용액 {v_target:.3f} mL"
                )
        self.log("="*40)


    def finish_ui_reset(self):
        self.btn_run.setEnabled(True)
        self.btn_snip.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font = app.font()
    font.setFamily("Malgun Gothic")
    app.setFont(font)
    
    ex = CellposeGUI()
    ex.show()
    sys.exit(app.exec())
