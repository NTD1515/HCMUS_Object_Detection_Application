"""CXR Disease Detection - Streamlit App.

Run: streamlit run app.py
"""

from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import RTDETR, YOLO


@dataclass(frozen=True)
class AppConfig:
    class_names: Tuple[str, ...] = (
        "Aortic enlargement",
        "Cardiomegaly",
        "Pleural effusion",
        "Pulmonary fibrosis",
        "Nodule/Mass",
    )
    yolo_run_dir: str = "Data/YOLO_Runs"
    yolo_run_prefix: str = "yolov8n_cxr_test"
    rtdetr_run_dir: str = "Data/Transformer_Runs"
    rtdetr_run_prefix: str = "rtdetr_cxr_test"
    faster_rcnn_weights: str = "Data/FasterRCNN_Runs/best_faster_rcnn.pth"
    img_size: int = 512
    default_conf: float = 0.25


CONFIG = AppConfig()


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (231, 76, 60),
    1: (52, 152, 219),
    2: (46, 204, 113),
    3: (241, 196, 15),
    4: (155, 89, 182),
}


def find_ultralytics_weights(run_dir: str, prefix: str) -> Optional[str]:
    """Return newest best.pt path; fall back to last.pt; None if missing."""
    for name in ("best.pt", "last.pt"):
        matches = sorted(
            glob.glob(os.path.join(run_dir, f"{prefix}*", "weights", name)),
            key=os.path.getmtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


@st.cache_resource(show_spinner="Loading YOLOv8...")
def load_yolo_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


@st.cache_resource(show_spinner="Loading RT-DETR...")
def load_rtdetr_model(weights_path: str) -> RTDETR:
    return RTDETR(weights_path)


@st.cache_resource(show_spinner="Loading Faster R-CNN...")
def load_faster_rcnn_model(
    weights_path: str,
    num_classes: int,
    device_str: str,
) -> torch.nn.Module:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Faster R-CNN weights not found: {weights_path}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(weights_path, map_location=device_str, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device_str).eval()
    return model


class Detector:
    """Unified inference interface for the three models."""

    def __init__(
        self,
        model_name: str,
        config: AppConfig,
        device: torch.device,
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.device = device

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
    ) -> pd.DataFrame:
        if self.model_name == "YOLOv8":
            return self._predict_yolo(image_rgb, conf_threshold)
        if self.model_name == "RT-DETR":
            return self._predict_rtdetr(image_rgb, conf_threshold)
        if self.model_name == "Faster R-CNN":
            return self._predict_faster_rcnn(image_rgb, conf_threshold)
        raise ValueError(f"Unsupported model: {self.model_name}")

    def _predict_yolo(self, image_rgb: np.ndarray, conf: float) -> pd.DataFrame:
        weights = find_ultralytics_weights(
            self.config.yolo_run_dir, self.config.yolo_run_prefix
        )
        if weights is None:
            raise FileNotFoundError(
                f"No YOLO weights found in "
                f"{self.config.yolo_run_dir}/{self.config.yolo_run_prefix}*/weights/"
            )
        model = load_yolo_model(weights)
        results = model.predict(source=image_rgb, conf=conf, verbose=False)
        return self._ultralytics_to_dataframe(results[0])

    def _predict_rtdetr(self, image_rgb: np.ndarray, conf: float) -> pd.DataFrame:
        weights = find_ultralytics_weights(
            self.config.rtdetr_run_dir, self.config.rtdetr_run_prefix
        )
        if weights is None:
            raise FileNotFoundError(
                f"No RT-DETR weights found in "
                f"{self.config.rtdetr_run_dir}/{self.config.rtdetr_run_prefix}*/weights/"
            )
        model = load_rtdetr_model(weights)
        results = model.predict(source=image_rgb, conf=conf, verbose=False)
        return self._ultralytics_to_dataframe(results[0])

    def _predict_faster_rcnn(
        self, image_rgb: np.ndarray, conf: float
    ) -> pd.DataFrame:
        model = load_faster_rcnn_model(
            self.config.faster_rcnn_weights,
            num_classes=len(self.config.class_names) + 1,
            device_str=str(self.device),
        )

        orig_h, orig_w = image_rgb.shape[:2]
        img_resized = cv2.resize(
            image_rgb,
            (self.config.img_size, self.config.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(img_norm).permute(2, 0, 1).contiguous().to(self.device)
        )

        with torch.no_grad():
            predictions = model([img_tensor])

        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        keep = scores >= conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) == 0:
            return self._empty_dataframe()

        # Scale bboxes back from img_size to original image dimensions
        scale_x = orig_w / self.config.img_size
        scale_y = orig_h / self.config.img_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Subtract 1 because class_id 0 is reserved for background
        class_ids = (labels - 1).astype(int)

        return pd.DataFrame({
            "class_id": class_ids,
            "class_name": [self.config.class_names[i] for i in class_ids],
            "x_min": boxes[:, 0].round(1),
            "y_min": boxes[:, 1].round(1),
            "x_max": boxes[:, 2].round(1),
            "y_max": boxes[:, 3].round(1),
            "confidence": scores.round(4),
        })

    def _ultralytics_to_dataframe(self, result) -> pd.DataFrame:
        if result.boxes is None or len(result.boxes) == 0:
            return self._empty_dataframe()

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return pd.DataFrame({
            "class_id": class_ids,
            "class_name": [self.config.class_names[i] for i in class_ids],
            "x_min": boxes[:, 0].round(1),
            "y_min": boxes[:, 1].round(1),
            "x_max": boxes[:, 2].round(1),
            "y_max": boxes[:, 3].round(1),
            "confidence": scores.round(4),
        })

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "class_id", "class_name",
                "x_min", "y_min", "x_max", "y_max",
                "confidence",
            ]
        )


class Visualizer:
    """Draws bounding boxes with labels on an image."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def draw(
        self,
        image_rgb: np.ndarray,
        detections: pd.DataFrame,
    ) -> np.ndarray:
        canvas = image_rgb.copy()
        h, w = canvas.shape[:2]
        thickness = max(2, int(round(min(h, w) / 400)))
        font_scale = max(0.5, min(h, w) / 1200.0)

        for _, row in detections.iterrows():
            class_id = int(row["class_id"])
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            x_min, y_min = int(row["x_min"]), int(row["y_min"])
            x_max, y_max = int(row["x_max"]), int(row["y_max"])
            label = f"{row['class_name']} {row['confidence']:.2f}"

            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, thickness)

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_y_top = max(y_min, text_h + baseline + 4)
            cv2.rectangle(
                canvas,
                (x_min, label_y_top - text_h - baseline - 4),
                (x_min + text_w + 4, label_y_top),
                color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                label,
                (x_min + 2, label_y_top - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )
        return canvas


class StreamlitApp:
    """Main UI orchestrator."""

    MODEL_OPTIONS: Tuple[str, ...] = ("YOLOv8", "Faster R-CNN", "RT-DETR")

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = Visualizer(config)

    def run(self) -> None:
        self._setup_page()
        self._init_session_state()
        model_name, conf_threshold = self._render_sidebar()
        self._render_main(model_name, conf_threshold)

    def _setup_page(self) -> None:
        st.set_page_config(
            page_title="CXR Disease Detection",
            page_icon="🩻",
            layout="wide",
        )
        st.title("🩻 Chest X-ray Disease Detection")
        st.caption("YOLOv8 · Faster R-CNN · RT-DETR | HCMUS Final Project")

    def _init_session_state(self) -> None:
        st.session_state.setdefault("results", None)
        st.session_state.setdefault("image_key", None)

    def _render_sidebar(self) -> Tuple[str, float]:
        st.sidebar.header("⚙️ Settings")
        model_name = st.sidebar.selectbox(
            "Model",
            options=self.MODEL_OPTIONS,
            index=0,
        )
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=self.config.default_conf,
            step=0.05,
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Device:** `{self.device}`")

        with st.sidebar.expander("📦 Weights status", expanded=False):
            self._render_weights_status()

        st.sidebar.markdown("**Disease classes:**")
        for i, name in enumerate(self.config.class_names):
            color = CLASS_COLORS.get(i, (0, 255, 0))
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            st.sidebar.markdown(
                f"<span style='color:{hex_color}; font-size: 1.2em;'>●</span> {name}",
                unsafe_allow_html=True,
            )
        return model_name, conf_threshold

    def _render_weights_status(self) -> None:
        # YOLOv8
        yolo_path = find_ultralytics_weights(
            self.config.yolo_run_dir, self.config.yolo_run_prefix
        )
        if yolo_path:
            run_name = os.path.basename(os.path.dirname(os.path.dirname(yolo_path)))
            st.success(f"YOLOv8: `{run_name}/{os.path.basename(yolo_path)}`")
        else:
            st.error("YOLOv8: weights not found")

        # RT-DETR
        rtdetr_path = find_ultralytics_weights(
            self.config.rtdetr_run_dir, self.config.rtdetr_run_prefix
        )
        if rtdetr_path:
            run_name = os.path.basename(os.path.dirname(os.path.dirname(rtdetr_path)))
            st.success(f"RT-DETR: `{run_name}/{os.path.basename(rtdetr_path)}`")
        else:
            st.error("RT-DETR: weights not found")

        # Faster R-CNN
        if Path(self.config.faster_rcnn_weights).exists():
            st.success(
                f"Faster R-CNN: `{os.path.basename(self.config.faster_rcnn_weights)}`"
            )
        else:
            st.error("Faster R-CNN: weights not found")

    def _render_main(self, model_name: str, conf_threshold: float) -> None:
        uploaded_file = st.file_uploader(
            "📤 Upload a chest X-ray image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is None:
            st.info("👆 Please upload a chest X-ray image to begin.")
            return

        image_pil = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image_pil)

        # Reset cached results when a different image is uploaded
        current_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.image_key != current_key:
            st.session_state.image_key = current_key
            st.session_state.results = None

        detect_clicked = st.button(
            "🔍 Detect Diseases",
            type="primary",
            use_container_width=True,
        )

        if detect_clicked:
            with st.spinner(f"Running inference with {model_name}..."):
                try:
                    detector = Detector(model_name, self.config, self.device)
                    detections = detector.predict(image_rgb, conf_threshold)
                    annotated = self.visualizer.draw(image_rgb, detections)
                    st.session_state.results = {
                        "annotated": annotated,
                        "detections": detections,
                        "model_name": model_name,
                        "conf_threshold": conf_threshold,
                    }
                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                    return
                except Exception as e:
                    st.error(f"❌ Inference error: {type(e).__name__}: {e}")
                    return

        if st.session_state.results is None:
            st.image(image_rgb, caption="Uploaded image", use_container_width=True)
            return

        r = st.session_state.results
        # Warn if user changed settings after the cached detection ran
        if r["model_name"] != model_name or r["conf_threshold"] != conf_threshold:
            st.warning(
                "Settings have changed since the last detection. "
                "Click **Detect Diseases** again to refresh."
            )

        self._display_results(
            image_rgb, r["annotated"], r["detections"], r["model_name"]
        )

    def _display_results(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        detections: pd.DataFrame,
        model_name: str,
    ) -> None:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original")
            st.image(original, use_container_width=True)
        with col_right:
            st.subheader(f"Detection ({model_name})")
            st.image(annotated, use_container_width=True)

        st.markdown("---")

        if detections.empty:
            st.warning(
                "⚠️ No detections at the current confidence threshold. "
                "Try lowering it in the sidebar and run detection again."
            )
            return

        st.subheader(f"📊 Statistics – {len(detections)} detection(s)")
        counts_per_class = detections["class_name"].value_counts()
        metric_cols = st.columns(len(self.config.class_names))
        for col, cls_name in zip(metric_cols, self.config.class_names):
            count = int(counts_per_class.get(cls_name, 0))
            col.metric(label=cls_name, value=count)

        detections_sorted = detections.sort_values(
            "confidence", ascending=False
        ).reset_index(drop=True)
        st.dataframe(detections_sorted, use_container_width=True)

        self._render_export_section(annotated, detections_sorted, model_name)

    def _render_export_section(
        self,
        annotated: np.ndarray,
        detections: pd.DataFrame,
        model_name: str,
    ) -> None:
        st.markdown("### 💾 Export")
        col_img, col_csv = st.columns(2)

        with col_img:
            img_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            success, png_buffer = cv2.imencode(".png", img_bgr)
            if success:
                st.download_button(
                    label="⬇️ Download annotated image (.png)",
                    data=png_buffer.tobytes(),
                    file_name=f"detection_{model_name.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with col_csv:
            csv_buffer = io.StringIO()
            detections.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download results (.csv)",
                data=csv_buffer.getvalue().encode("utf-8-sig"),
                file_name=f"detection_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main() -> None:
    app = StreamlitApp(CONFIG)
    app.run()


if __name__ == "__main__":
    main()
