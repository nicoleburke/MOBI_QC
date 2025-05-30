import sys

print(sys.executable)
print(sys.path)

import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
    QPushButton, QLabel, QScrollArea, QGroupBox, QTextBrowser
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap

modality_keywords = {
    "EEG": "eeg",
    "Eye Tracking": "_et_",
    "ECG": "ecg",
    "EDA": "eda",
    "RSP": "rsp",
    "Graphomotor": "graphomotor",
    "Audio": "mic",
    "Video": "webcam"
}

modality_texts = {
    "EEG": """
<b>Bad channels before robust reference:</b><br>
['E84', 'E74', 'E7', 'E126', 'E82', 'E94', 'E81', 'E128', 'E106', 'E83', 'E76', 'Cz', 'E89', 'E77', 'E90', 'E88', 'E95']<br>
<b>Interpolated channels:</b><br>
['E84', 'E74', 'E126', 'E128', 'E94', 'E73', 'E81', 'E25', 'E127', 'Cz', 'E89', 'E8', 'E77', 'E90', 'E88', 'E95']<br>
<b>Bad channels after interpolation:</b><br>
['E63', 'E82']<br>
<b>Median Peak-to-peak Amplitude:</b> 0.00032130 volts<br>
<b>Median Standard Deviation of Amplitude:</b> 2.11e-5<br>
<b>Percent Muscle/Eye Artifacts:</b> 10.36 %<br>
<b>Percent Good:</b> 89.64 %<br>
<b>Mean power at 60 Hz:</b> 5.0074e-13 V²/Hz<br>
<b>Mean SNR:</b> 116.76 dB<br>
""",
    "Eye Tracking": """
<b>Effective Sampling Rate:</b> 499.827 Hz<br>
<b>Flag 1:</b> all coordinates have the same % validity: Yes<br>
<b>Flag 2:</b> % of NaNs same across UCS/TBCS & display area: Yes<br>
<b>Mean difference in percent valid data:</b> 1.70%<br>
<b>Gaze point diff over 0.2 mm:</b> 6.93%<br>
""",
    "ECG": """
<b>Effective sampling rate:</b> 499.763 Hz<br>
<b>SNR:</b> 10.328 dB<br>
<b>Average Heart Rate:</b> 78.073 bpm<br>
<b>kSQI:</b> 16.529<br>
<b>pSQI:</b> 0.787 mV²/Hz<br>
<b>basSQI:</b> 26.979%<br>
""",
    "EDA": """
<b>Effective Sampling Rate:</b> 499.763 Hz<br>
<b>Signal Integrity Check:</b> 100%<br>
<b>SNR:</b> 52.325 dB<br>
<b>Avg SCL:</b> 6.721 mS<br>
<b>SCL Std Dev:</b> 0.974 mS<br>
<b>SCL CoV:</b> 14.490%<br>
<b>Avg SCR Amplitude:</b> 0.251 mS<br>
<b>SCR Amplitude Validity:</b> 96.678%<br>
""",
    "RSP": """
<b>Sampling Rate:</b> 499.938 Hz<br>
<b>SNR:</b> 9.307 dB<br>
<b>Breath Amplitude Mean:</b> 0.283 V<br>
<b>Breath Amp Std Dev:</b> 0.384 V<br>
<b>Breath Amp Range:</b> 0.0837 - 2.673 V<br>
<b>Respiration Rate Mean:</b> 19.387 bpm<br>
<b>Resp Rate Std Dev:</b> 2.938 bpm<br>
<b>Resp Rate Range:</b> 9.837 - 26.847 bpm<br>
<b>Peak Interval Mean:</b> 2.99 sec<br>
<b>Peak Interval Std Dev:</b> 1.476 sec<br>
<b>Peak Interval Range:</b> 0.733 - 12.842 sec<br>
<b>Baseline Drift:</b> 0.0587 V<br>
<b>Autocorrelation:</b> -0.06<br>
""",
    "Audio": """
<b>Duration Difference:</b> 0 sec<br>
<b>Sampling Rate:</b> 44098.176 Hz<br>
<b>Percent NaNs:</b> 0%<br>
<b>1st Quartile:</b> -37.0<br>
<b>3rd Quartile:</b> 37.0<br>
<b>Mean:</b> 0.2<br>
<b>Std Dev:</b> 157.32<br>
<b>Range:</b> -5742 - 6074<br>
""",
    "Video": """
<b>Sampling Rate:</b> 30.04 Hz<br>
<b>Duration Match:</b> True<br>
<b>% of frames with face detected:</b> 100%<br>
"""
}

class ModalityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modality QC Viewer")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left Panel
        modality_panel = QGroupBox("Modality Panel")
        modality_layout = QVBoxLayout()

        self.checkboxes = {}
        for modality in modality_keywords.keys():
            checkbox = QCheckBox(modality)
            checkbox.stateChanged.connect(self.display_qc_output)
            self.checkboxes[modality] = checkbox
            modality_layout.addWidget(checkbox)

        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all)

        modality_layout.addWidget(select_all_button)
        modality_layout.addWidget(deselect_all_button)
        modality_panel.setLayout(modality_layout)

        # Right Panel
        self.output_browser = QTextBrowser()
        self.output_browser.setOpenExternalLinks(True)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.output_browser)

        main_layout.addWidget(modality_panel, 1)
        main_layout.addWidget(scroll_area, 3)
        self.setLayout(main_layout)

    def display_qc_output(self):
        self.output_browser.clear()
        selected = [m for m, cb in self.checkboxes.items() if cb.isChecked()]
        for modality in selected:
            title = f"<br><center><span style='font-size:20pt;'><b>{modality} QC Metrics</b></span></center><hr>"
            if modality in modality_texts:
                text = f"<div style='font-size:14pt;'>{modality_texts[modality]}</div>"
            elif modality == "Graphomotor":
                text = "<div style='font-size:14pt;'>No QC data available.</div>"
            else:
                text = "<div style='font-size:14pt;'>No QC data available.</div>"

            self.output_browser.append(title + text)

            folder = "images"
            keyword = modality_keywords[modality].lower()
            if os.path.exists(folder):
                for fname in sorted(os.listdir(folder)):
                    if keyword in fname.lower():
                        image_path = os.path.join(folder, fname)
                        self.output_browser.append(f'<img src="{image_path}" style="max-width:100%; height:auto;"></div>')

        self.output_browser.verticalScrollBar().setValue(0)

    def select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)
        self.display_qc_output()

    def deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)
        self.display_qc_output()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModalityApp()
    window.show()
    sys.exit(app.exec())