import io
import os
import random
from pathlib import Path
import gdown
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
MODEL_CANDIDATES = [
    ROOT_DIR / "vgg16_aug_tuned.keras",
]
DATASET_ROOT = ROOT_DIR / "datasets" / "datasets" / "Covid19-dataset"
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]
TARGET_SIZE = (256, 256)

# -----------------------------------------------------------------------------
# Google Drive Dataset Configuration
# -----------------------------------------------------------------------------

DATASET_ZIP_PATH = ROOT_DIR / "covid_dataset.zip"

# Replace with your actual Google Drive File ID
DATASET_FILE_ID = "17o7carABp1Isl8SoylQco5douKYpY9G2"

DATASET_URL = "https://drive.google.com/file/d/17o7carABp1Isl8SoylQco5douKYpY9G2/view?usp=sharing"
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def apply_custom_styles() -> None:
    """Add professional theme, spacing, and card styling."""
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 1.8rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stApp {
            background: linear-gradient(180deg, #061526 0%, #0f2340 100%);
            color: #e7eef7;
        }
        .stButton>button {
            background-color: #0b6e99;
            color: white;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #0f8cc9;
            color: white;
        }
        .metric-card {
            border-radius: 18px;
            padding: 1.2rem;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 15px 40px rgba(0,0,0,0.18);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #071a33 0%, #0b2d56 100%);
        }
        .footer {
            color: #a9b6d4;
            text-align: center;
            padding: 1rem 0;
            font-size: 0.95rem;
        }
        .model-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.4rem 0.9rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #e1eaff;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_resource():

    model_path = get_model_path()

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

def get_model_path() -> Path:
    """Detect the best available saved model file."""
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No saved model found."
    )

@st.cache_resource(show_spinner=True)
def download_and_extract_dataset():

    if DATASET_ROOT.exists():
        return

    try:
       
        url = DATASET_URL

        gdown.download(
            url,
            str(DATASET_ZIP_PATH),
            quiet=False
        )

        

        with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(ROOT_DIR / "datasets")

       

    except Exception as e:
        st.error(f"Dataset download failed: {e}")
        st.stop()
      
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load an uploaded image from bytes into a PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as error:
        raise ValueError("The uploaded file is not a supported image.") from error


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and format the image for model prediction."""
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def make_prediction(model: tf.keras.Model, image_array: np.ndarray) -> dict:
    """Run the model and return class probabilities and predicted label."""
    prediction = model.predict(image_array, verbose=0)[0]
    top_index = int(np.argmax(prediction))
    return {
        "label": CLASS_NAMES[top_index],
        "confidence": float(prediction[top_index]),
        "probabilities": prediction,
    }


def format_confidence_gauge(label: str, confidence: float) -> tuple[str, str]:
    """Return a message style and status for each predicted class."""
    if label == "Covid":
        return (f"High risk of COVID-19 detected ({confidence*100:.1f}%)", "error")
    if label == "Viral Pneumonia":
        return (f"Possible pneumonia detected ({confidence*100:.1f}%)", "warning")
    return (f"Normal chest X-ray detected ({confidence*100:.1f}%)", "success")


@st.cache_data(show_spinner=False)
def scan_dataset() -> dict:
    """Scan local dataset directories and return class counts."""
    summary = {
        "train": {},
        "test": {},
    }
    for split in ["train", "test"]:
        split_folder = DATASET_ROOT / split
        if not split_folder.exists():
            continue
        for class_name in CLASS_NAMES:
            class_path = split_folder / class_name
            count = 0
            if class_path.exists():
                count += sum(1 for _ in class_path.rglob("*.png"))
                count += sum(1 for _ in class_path.rglob("*.jpg"))
                summary[split][class_name] = count
        for class_path in split_folder.iterdir() if split_folder.exists() else []:
            if class_path.is_dir() and class_path.name not in summary[split]:
                summary[split][class_path.name] = (
                    sum(1 for _ in class_path.rglob("*.png")) + sum(1 for _ in class_path.rglob("*.jpg"))
                )
    return summary


@st.cache_data(show_spinner=False)
def collect_sample_images(limit: int = 6) -> list[tuple[str, Image.Image]]:
    """Collect a small set of sample images from the dataset for display."""
    sample_images = []
    if not DATASET_ROOT.exists():
        return sample_images
    sample_paths = []
    for split in ["train", "test"]:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue
        for cls in split_dir.iterdir():
            if cls.is_dir():
                sample_paths.extend([p for p in cls.glob("*.png")])
                sample_paths.extend([p for p in cls.glob("*.jpg")])
    random.shuffle(sample_paths)
    for path in sample_paths[:limit]:
        try:
            image = Image.open(path).convert("RGB")
            sample_images.append((path.name, image))
        except Exception:
            continue
    return sample_images


def generate_probability_chart(probabilities: np.ndarray) -> go.Figure:
    """Build a horizontal bar chart for class probability distribution."""
    df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities * 100})
    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        text="Probability",
        color="Class",
        color_discrete_sequence=["#f45d48", "#42c4bc", "#f4d35e"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Confidence (%)",
        yaxis_title="Prediction",
        template="plotly_dark",
        height=320,
    )
    return fig


@st.cache_data(show_spinner=False)
def evaluate_model_on_test() -> dict:
    """Evaluate the loaded model against the local test dataset for performance metrics."""
    model = load_model_resource()
    results = {
        "y_true": [],
        "y_pred": [],
        "y_score": [],
        "labels": CLASS_NAMES,
        "summary": {},
    }
    test_root = DATASET_ROOT / "test"
    if not test_root.exists():
        return results
    for idx, class_name in enumerate(CLASS_NAMES):
        class_folder = test_root / class_name
        if not class_folder.exists():
            continue
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for image_path in class_folder.glob(ext):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_array = preprocess_image(image)
                    prediction = model.predict(image_array, verbose=0)[0]
                    predicted_label = int(np.argmax(prediction))
                    results["y_true"].append(idx)
                    results["y_pred"].append(predicted_label)
                    results["y_score"].append(prediction)
                except Exception:
                    continue
    if not results["y_true"]:
        return results
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])
    y_score = np.vstack(results["y_score"])
    results["summary"] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        if y_score.shape[1] == len(CLASS_NAMES):
            results["summary"]["roc_auc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
        else:
            results["summary"]["roc_auc"] = None
    except Exception:
        results["summary"]["roc_auc"] = None
    return results


def create_prediction_dataframe(prediction_records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame for prediction history and download."""
    return pd.DataFrame(
        [
            {
                "Image": record["filename"],
                "Prediction": record["label"],
                "Confidence": f"{record['confidence']*100:.2f}%",
            }
            for record in prediction_records
        ]
    )


def display_footer() -> None:
    footer_html = """
    <div class='footer'>
        Built with TensorFlow · Keras · Streamlit · Python<br/>
        <span style='opacity: 0.78;'>Use this app to explore saved CNN inference for COVID-19 chest X-ray classification.</span>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def render_home_page(model_path: Path) -> None:
    st.subheader("COVID-19 X-Ray Detection Dashboard")
    st.markdown(
        """
        **A professional inference app for CNN-based chest X-ray classification.**
        Upload a sample and see how the trained model detects COVID, Normal or Pneumonia cases instantly.
        """
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Why this app helps clinicians")
        st.write(
            "- Fast inference for chest X-ray screening.\n"
            "- Clear probability outputs and risk indicators.\n"
            "- Simple, polished interface for deployment and reporting."
        )
        st.markdown("### Key highlights")
        st.write(
            "- Uses the best available saved model for inference.\n"
            "- Multi-class prediction for COVID, Normal lungs, and Pneumonia.\n"
            "- Lightweight UI optimized for diagnostic workflows."
        )
    with right:
        st.image(
            "https://images.unsplash.com/photo-1586773860415-7bbfda9657ef?auto=format&fit=crop&w=900&q=80",
            caption="Chest X-ray classification for rapid screening",
            use_column_width=True,
        )

    st.markdown("---")

    metrics = st.columns(4)
    with metrics[0]:
        st.metric(label="Selected Model", value="VGG-16")
    with metrics[1]:
        st.metric(label="Target Classes", value=len(CLASS_NAMES))
    with metrics[2]:
        st.metric(label="Inference Mode", value="GPU/CPU Ready")
    with metrics[3]:
        st.metric(label="Deployment", value="Streamlit")

    with st.expander("Technology Stack and Model Details"):
        st.markdown(
            """
            - TensorFlow / Keras for model inference.
            - Streamlit for UI and dashboard presentation.
            - PIL + NumPy for image preprocessing.
            - Plotly for interactive probability charts.
            """
        )

    display_footer()


def render_about_page() -> None:
    st.title("About the Project")
    st.markdown(
        """
        ### Problem Statement
        Pneumonia and COVID-19 are respiratory conditions that can be detected by changes in a chest X-ray. Manual diagnosis requires time and radiological expertise. This project builds an automated CNN-powered analysis pipeline to help accelerate screening.
        """
    )
    st.markdown("### Objective")
    st.write(
        "Use a trained Convolutional Neural Network to classify chest X-ray images into COVID-19, Normal, or Viral Pneumonia categories. "
        "The system is designed for inference only and uses the best locally saved model artifact."
    )
    st.markdown("### Why CNN?")
    st.write(
        "Convolutional Neural Networks extract image features directly from X-ray scans, capturing texture, edges, and region patterns. "
        "Their ability to learn hierarchical representations makes them ideal for medical imaging tasks such as lung abnormality detection."
    )
    st.markdown("### Deep Learning Workflow")
    st.write(
        "1. Image collection and preprocessing.\n"
        "2. CNN architecture design and training offline.\n"
        "3. Model validation and saving.\n"
        "4. Deployment via this Streamlit inference dashboard."
    )
    st.markdown("### Medical Importance")
    st.write(
        "Early identification of COVID-19 and pneumonia can support triage, reduce radiologist workload, and improve clinical response. "
        "This app is built for inference, not training, to preserve compute resources and maintain performance consistency."
    )
    display_footer()


def render_dataset_page() -> None:
    st.title("Dataset Overview")
    st.write(
        "This section summarizes the structure and distribution of the chest X-ray dataset used for inference evaluation."
    )
    summary = scan_dataset()
    st.subheader("Dataset Structure")
    if summary["train"] or summary["test"]:
        st.write("The dataset is organized into train/test splits with class folders for each label.")
        st.json({"train": summary["train"], "test": summary["test"]})
    else:
        st.warning("Dataset folder not available locally. Place `datasets/Covid19-dataset` to enable dataset visualization.")

    st.subheader("Sample Images")
    samples = collect_sample_images(limit=6)
    if samples:
        grid = st.columns(3)
        for idx, (name, image) in enumerate(samples):
            with grid[idx % 3]:
                st.image(image, caption=name, use_column_width=True)
    else:
        st.info("No sample images found in the dataset folder.")

    st.subheader("Preprocessing Pipeline")
    st.markdown(
        """
        - Resize images to `256x256` pixels.
        - Normalize pixel values to the `[0, 1]` range.
        - Convert images to RGB format.
        - Expand dimensions for model batch input.
        """
    )
    st.subheader("Training Augmentation Techniques")
    st.markdown(
        """
        The model was trained offline with augmentation to improve generalization:
        - Random horizontal flips
        - Random rotations
        - Random zoom transformations
        - Dropout and regularization in the dense head
        """
    )
    display_footer()


def render_model_page() -> None:
    st.title("Model Information")
    model_path = get_model_path()
    st.write("Using the best available saved model file for inference.")
    st.markdown(f"**Model file:** `{model_path.name}`")
    model = load_model_resource()

    st.subheader("Model Architecture")
    with st.expander("View model summary"):
        buffer = io.StringIO()
        model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        st.text(buffer.getvalue())

    st.subheader("Technical Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Layers", len(model.layers))
        st.metric("Trainable Parameters", int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])))
    with col2:
        st.metric("Non-trainable Params", int(np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])))
        st.metric("Input Shape", str(model.input_shape))

    st.markdown("### Core CNN Concepts")
    st.markdown(
        "- Convolutional layers to capture spatial features.\n"
        "- Pooling layers to reduce dimensionality.\n"
        "- Dense classification head with softmax activation.\n"
        "- Transfer learning base if the model includes a pre-trained backbone."
    )
    st.markdown("### Optimization Details")
    st.markdown(
        "- Loss function: `sparse_categorical_crossentropy`\n"
        "- Optimizer: `Adam` (offline training)\n"
        "- Output: 3-way softmax classification\n"
        "- Classes: COVID, Normal, Viral Pneumonia"
    )
    st.markdown("### Notes")
    st.write(
        "This application focuses on inference only and does not include any training code. "
        "The model was trained offline and loaded directly from saved artifacts."
    )
    display_footer()


def render_prediction_page() -> None:
    st.title("Prediction")
    st.write("Upload a chest X-ray and get a fast inference with probability scores.")

    model = load_model_resource()
    prediction_history = []

    with st.expander("Upload Options"):
        uploaded_files = st.file_uploader(
            "Choose one or more X-ray images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if st.button("Use webcam image"):
            st.info("Camera input is available in the browser. Please use a supported device.")
        camera_image = st.camera_input("Take a live X-ray style photo")

    images_to_predict = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                pil_image = load_image_from_bytes(uploaded_file.read())
                images_to_predict.append((uploaded_file.name, pil_image))
            except ValueError as error:
                st.error(str(error))
    elif camera_image is not None:
        try:
            pil_image = load_image_from_bytes(camera_image.getvalue())
            images_to_predict.append(("camera_image.jpg", pil_image))
        except ValueError as error:
            st.error(str(error))

    if not images_to_predict:
        st.info("Upload a chest X-ray image above to see model predictions.")
        display_footer()
        return

    for filename, image in images_to_predict:
        st.markdown(f"### {filename}")
        cols = st.columns([2, 3])
        with cols[0]:
            st.image(image, caption="Uploaded image", use_column_width=True)
        with cols[1]:
            image_array = preprocess_image(image)
            with st.spinner("Running model inference..."):
                prediction = make_prediction(model, image_array)
            message, status = format_confidence_gauge(prediction["label"], prediction["confidence"])
            if status == "success":
                st.success(message)
            elif status == "warning":
                st.warning(message)
            else:
                st.error(message)
            st.markdown("**Prediction details**")
            st.write(
                f"- Predicted label: **{prediction['label']}**\n"
                f"- Confidence: **{prediction['confidence']*100:.2f}%**"
            )
            st.plotly_chart(generate_probability_chart(prediction["probabilities"]), use_container_width=True)
            prediction_history.append({
                "filename": filename,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            })

    if prediction_history:
        st.markdown("---")
        st.subheader("Prediction History")
        history_df = create_prediction_dataframe(prediction_history)
        st.dataframe(history_df, use_container_width=True)
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction report",
            data=csv_data,
            file_name="prediction_report.csv",
            mime="text/csv",
        )
    display_footer()


def render_performance_page() -> None:
    st.title("Model Performance")
    st.write("Evaluation results from the available test dataset and inference metrics.")
    metrics = evaluate_model_on_test()
    summary = metrics.get("summary", {})
    if not summary:
        st.warning("No test dataset was found locally, so performance metrics cannot be computed.")
        display_footer()
        return

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{summary['accuracy']*100:.2f}%")
    cols[1].metric("Precision", f"{summary['precision_macro']*100:.2f}%")
    cols[2].metric("Recall", f"{summary['recall_macro']*100:.2f}%")
    cols[3].metric("F1 Score", f"{summary['f1_macro']*100:.2f}%")
    if summary.get("roc_auc") is not None:
        st.metric("ROC-AUC", f"{summary['roc_auc']*100:.2f}%")

    st.markdown("### Confusion Matrix")
    cm = np.array(summary["confusion_matrix"])
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Viridis",
            hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if metrics["y_true"]:
        st.markdown("### Prediction Distribution")
        chart_df = pd.DataFrame({
            "Actual": metrics["y_true"],
            "Predicted": metrics["y_pred"],
        })
        chart_df["Actual"] = chart_df["Actual"].map(lambda x: CLASS_NAMES[x])
        chart_df["Predicted"] = chart_df["Predicted"].map(lambda x: CLASS_NAMES[x])
        fig = px.histogram(
            chart_df,
            x="Actual",
            color="Predicted",
            barmode="group",
            title="Actual vs Predicted Class Distribution",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        > Note: Training curves are not included in the local saved artifacts. This dashboard focuses only on inference and evaluation from available saved models and test images.
        """
    )
    display_footer()


def get_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if hasattr(layer, "layers"):

            for nested_layer in reversed(layer.layers):

                if isinstance(nested_layer, tf.keras.layers.Conv2D):
                    return nested_layer.name

    return None


def make_gradcam_heatmap(model, image_array, last_conv_layer_name):

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(image_array)

            predicted_index = tf.argmax(predictions[0])

            loss = predictions[:, predicted_index]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except Exception as e:

        st.error(f"Grad-CAM failed: {e}")

        return np.zeros((10, 10))


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay the Grad-CAM heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(image.size).convert("RGBA")
    heatmap_img = Image.blend(image.convert("RGBA"), heatmap_img, alpha=alpha)
    return heatmap_img


def render_visualization_page() -> None:
    st.title("Visualizations")
    st.write("Explore predictions, sample behavior, and activation insights.")
    sample_images = collect_sample_images(limit=4)
    if sample_images:
        st.subheader("Sample dataset images")
        cols = st.columns(4)
        for idx, (name, image) in enumerate(sample_images):
            with cols[idx]:
                st.image(image, caption=name, use_column_width=True)
    else:
        st.info("No dataset images were found for sample display.")

    st.subheader("Sample Predictions")
    model = load_model_resource()
    if sample_images:
        rows = st.columns(2)
        for idx, (name, image) in enumerate(sample_images[:2]):
            with rows[idx]:
                st.image(image, caption=f"Sample {idx+1}", use_column_width=True)
                array = preprocess_image(image)
                prediction = make_prediction(model, array)
                st.markdown(f"**Prediction:** {prediction['label']}  ")
                st.markdown(f"**Confidence:** {prediction['confidence']*100:.2f}%")
    else:
        st.write("Upload images on the Prediction page to see sample inference results.")

    last_conv = get_last_conv_layer(model)
    if last_conv is not None and sample_images:
        st.subheader("Grad-CAM Activation Map")
        image_name, image = sample_images[0]
        image_array = preprocess_image(image)
        heatmap = make_gradcam_heatmap(model, image_array, last_conv)
        heatmap_overlay = overlay_heatmap(image, heatmap)
        cols = st.columns(2)
        with cols[0]:
            st.image(image, caption="Input image", use_column_width=True)
        with cols[1]:
            st.image(heatmap_overlay, caption="Grad-CAM overlay", use_column_width=True)
        st.markdown(
            "This visualization highlights the regions that most influenced the network's decision."
        )
    else:
        st.info("Grad-CAM is unavailable because the model architecture has no detectable convolutional layer or dataset images are unavailable.")
    display_footer()


def render_contact_page() -> None:
    st.title("Contact & Notes")
    st.write("Use this dashboard to share results, export reports, and support rapid image-based screening.")
    st.markdown("**Project Folder:** `Covid_19`")
    st.markdown("**Saved Model:** VGG-16 Augmentated and Hypertuned \n**Notebook:** `Richa_K_Batch_13_ANN_CNN_Mini_Project.ipynb`")
    st.markdown(
        "If you need to deploy this app to Streamlit Cloud, run `streamlit run app.py` from the project root."
    )
    st.markdown("### Development notes")
    st.write(
        "- The app only performs inference with existing saved models.\n"
        "- No training code is included.\n"
        "- It supports batch uploads, live camera capture, and downloadable reports."
    )
    display_footer()


def main() -> None:
    st.set_page_config(
        page_title="COVID-19 X-Ray Inference",
        page_icon="🩺",
        layout="wide",
    )
    apply_custom_styles()
    download_and_extract_dataset()
    st.sidebar.title("COVID Inference App")
    st.sidebar.markdown("A polished dashboard for CNN-based chest X-ray screening.")
    pages = [
        "Home",
        "About Project",
        "Dataset Overview",
        "Model Information",
        "Prediction",
        "Model Performance",
        "Visualizations",
        "Contact",
    ]
    selected_page = st.sidebar.radio("Navigate", pages)

    sidebar_footer = st.sidebar.empty()
    sidebar_footer.markdown(
        "---\n" 
        "Built with Streamlit · TensorFlow · Keras"
    )

    model_path = None
    try:
        model_path = get_model_path()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    if selected_page == "Home":
        render_home_page(model_path)
    elif selected_page == "About Project":
        render_about_page()
    elif selected_page == "Dataset Overview":
        render_dataset_page()
    elif selected_page == "Model Information":
        render_model_page()
    elif selected_page == "Prediction":
        render_prediction_page()
    elif selected_page == "Model Performance":
        render_performance_page()
    elif selected_page == "Visualizations":
        render_visualization_page()
    elif selected_page == "Contact":
        render_contact_page()


if __name__ == "__main__":
    main()
