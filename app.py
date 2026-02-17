import streamlit as st

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import json
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import io
import torch
from torchvision.ops import nms
import psutil
import time
import os
import cv2
import plotly.express as px
import hashlib
from neo4j import GraphDatabase

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_REGEX = r"(\d{2})?\s*([A-Z]{2,5})\s*(\d{4})"
FONT_PATH = None
Image.MAX_IMAGE_PIXELS = None  # Allow large images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_EMBEDDED = os.environ.get("BSG_EMBEDDED") == "1"
CLASS_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state["model"] = None  # YOLO model
if "all_images" not in st.session_state:
    st.session_state["all_images"] = []
if "all_image_names" not in st.session_state:
    st.session_state["all_image_names"] = []
if "raw_detections" not in st.session_state:
    st.session_state["raw_detections"] = None
if "processed_images" not in st.session_state:
    st.session_state["processed_images"] = []
if "current_image_index" not in st.session_state:
    st.session_state["current_image_index"] = 0

if "initial_mem" not in st.session_state:
    proc = psutil.Process(os.getpid())
    st.session_state["initial_mem"] = proc.memory_info().rss / (1024 * 1024)
    st.session_state["start_clock"] = time.time()

if "res_usage" not in st.session_state:
    st.session_state["res_usage"] = {
        "pdf_time": 0.0,
        "pdf_mem_diff": 0.0,
        "detect_time": 0.0,
        "detect_mem_diff": 0.0,
        "ocr_time": 0.0,
        "ocr_mem_diff": 0.0,
    }

if "detection_config" not in st.session_state:
    st.session_state["detection_config"] = {
        "conf_threshold": 0.25,
        "grid_size": (4, 4),
        "nms_threshold": 0.5,
        "apply_nms": True,
        "max_grid_size": 4,
        "dynamic_grid": False,
        "inference_batch_size": 4,
        "inference_timeout": 60,
        "bbox_color": "#FF0000",
        "bbox_thickness": 3,
        "color_by_class": True,
    }

if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None
if "asset_locations" not in st.session_state:
    st.session_state["asset_locations"] = None

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS / CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    """Load the RapidOCR engine once."""
    return RapidOCR()

@st.cache_resource
def load_yolo_model(selected_model: str, device="cpu"):
    """Load YOLO model from a .pt file, cached for the session."""
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=selected_model,
        force_reload=False
    ).to(device)
    return model

# -----------------------------------------------------------------------------
# PDF -> Image Conversion
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def render_pdf_to_images(file_bytes: bytes, dpi=300, max_pages=50):
    """Convert PDF into a list of PIL images."""
    images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        if total_pages > max_pages:
            st.warning(f"PDF has {total_pages} pages; only the first {max_pages} processed.")
            doc = doc[:max_pages]

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('RGBA')
            images.append(img)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return images

def maybe_downscale(img: Image.Image, max_dim=3000) -> Image.Image:
    """Downscale large images to reduce memory usage & speed detection."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_size = (int(w*scale), int(h*scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

# -----------------------------------------------------------------------------
# IMAGE ENHANCEMENT
# -----------------------------------------------------------------------------
def enhance_images(
    images,
    resize_factor=1.0,
    denoise_strength=10,
    denoise_template_window_size=7,
    denoise_search_window=21,
    thresholding=False,
    deskew_angle=0
):
    enhanced = []
    for img in images:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

        # Resize if needed
        if resize_factor != 1.0:
            new_w = int(cv_img.shape[1] * resize_factor)
            new_h = int(cv_img.shape[0] * resize_factor)
            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Denoise
        cv_img = cv2.fastNlMeansDenoisingColored(
            cv_img,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=denoise_template_window_size,
            searchWindowSize=denoise_search_window
        )

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Threshold
        if thresholding:
            _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Deskew
        if deskew_angle != 0:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pil_enhanced = Image.fromarray(gray).convert("RGBA")
        enhanced.append(pil_enhanced)

    return enhanced

# -----------------------------------------------------------------------------
# DETECTION (GRID SPLITTING, INFERENCE, NMS)
# -----------------------------------------------------------------------------
def split_image_into_grid(image, grid_size=(4, 4)):
    width, height = image.size
    cols, rows = grid_size
    grid_images = []
    coordinates = []

    grid_width = width // cols
    grid_height = height // rows

    for i in range(cols):
        for j in range(rows):
            left = i * grid_width
            upper = j * grid_height
            right = min((i + 1) * grid_width, width)
            lower = min((j + 1) * grid_height, height)
            cropped_img = image.crop((left, upper, right, lower))
            grid_images.append(cropped_img)
            coordinates.append((left, upper, right, lower))

    return grid_images, coordinates

def run_inference_and_get_results(model, imgs, confidence_threshold, apply_nms=True, nms_threshold=0.5):
    """Perform YOLO inference on a list of PIL images."""
    model.conf = confidence_threshold
    results = model(imgs)  # YOLOv5 automatically handles PIL -> tensor

    batched_detections = []
    for output in results.xyxy:
        detected_objects = []
        if len(output) == 0:
            batched_detections.append(detected_objects)
            continue

        boxes = []
        scores = []
        for bbox in output.cpu().numpy():
            class_id = int(bbox[5])
            conf = bbox[4]
            if conf >= confidence_threshold:
                detected_objects.append({
                    "class": model.names[class_id],
                    "confidence": float(conf),
                    "bbox": [float(x) for x in bbox[:4]]
                })
                boxes.append(bbox[:4])
                scores.append(conf)

        # Apply NMS if requested
        if apply_nms and boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=DEVICE)
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE)
            keep = nms(boxes_tensor, scores_tensor, nms_threshold)
            keep = keep.cpu().numpy()  # back to CPU
            detected_objects = [detected_objects[i] for i in keep]

        batched_detections.append(detected_objects)

    return batched_detections

def map_detections_to_original(detections, grid_coord):
    (left_offset, top_offset, _, _) = grid_coord
    mapped = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        mapped_bbox = [
            xmin + left_offset,
            ymin + top_offset,
            xmax + left_offset,
            ymax + top_offset
        ]
        mapped.append({
            "class": det["class"],
            "confidence": det["confidence"],
            "bbox": mapped_bbox
        })
    return mapped

def dynamic_grid_size(image, max_grid=4):
    """Heuristic for deciding a dynamic grid size based on image area."""
    width, height = image.size
    area = width * height
    if area > 5_000_000:
        return (min(max_grid, 8), min(max_grid, 8))
    else:
        return (4, 4)

def detect_objects_in_grids(
    model,
    image,
    conf_threshold=0.25,
    grid_size=(4,4),
    nms_threshold=0.5,
    apply_nms=True,
    inference_batch_size=4,
    dynamic_grid=False,
    max_grid_size=4
):
    if dynamic_grid:
        grid_size = dynamic_grid_size(image, max_grid_size)

    grid_images, coords = split_image_into_grid(image, grid_size=grid_size)
    all_detections = []

    tile_count = len(grid_images)
    progress_bar = st.progress(0)

    for i in range(0, tile_count, inference_batch_size):
        batch_tiles = grid_images[i : i + inference_batch_size]
        batch_results = run_inference_and_get_results(
            model=model,
            imgs=batch_tiles,
            confidence_threshold=conf_threshold,
            apply_nms=apply_nms,
            nms_threshold=nms_threshold
        )
        for j, sub_res in enumerate(batch_results):
            grid_coord = coords[i + j]
            mapped = map_detections_to_original(sub_res, grid_coord)
            all_detections.extend(mapped)

        progress_val = int(((i + len(batch_tiles)) / tile_count) * 100)
        progress_bar.progress(progress_val)

    return all_detections

# -----------------------------------------------------------------------------
# OCR & TEXT EXTRACTION
# -----------------------------------------------------------------------------
def extract_text_with_ocr(ocr_engine, image):
    np_image = np.array(image)
    result, _ = ocr_engine(np_image)
    return " ".join([res[1] for res in result]) if result else ""

def normalize_text(text):
    text = re.sub(r'\b(?:Instrument|nstrument|mstrument)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_tagname_regex(custom_regex=None):
    return custom_regex if custom_regex else DEFAULT_REGEX

def extract_tagnames_from_text(text, regex_pattern, system_hint="13"):
    matches = re.findall(regex_pattern, text)
    extracted = []
    for m in matches:
        if len(m) >= 3:
            system_number = m[0] if m[0] else system_hint
            function_code = m[1]
            loop_sequence = m[2]
            extracted.append({
                "system_number": system_number,
                "function_code": function_code,
                "loop_sequence": loop_sequence
            })
    return extracted

def build_tagname(system_number, function_code, loop_sequence, parts, separators):
    value_map = {
        "System Number": system_number,
        "Function Code": function_code,
        "Loop Sequence": loop_sequence
    }
    final_str = ""
    used_count = 0
    for i in range(3):
        part = parts[i]
        if part != "None":
            if used_count > 0:
                final_str += separators[used_count - 1]
            final_str += value_map.get(part, "")
            used_count += 1
    return final_str

def color_for_class(class_name, palette=None):
    if not class_name:
        return "#FF0000"
    palette = palette or CLASS_COLOR_PALETTE
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(palette)
    return palette[idx]

def resolve_env_path(path: str):
    candidate_paths = []
    path_obj = Path(path)
    if path_obj.is_absolute():
        candidate_paths.append(path_obj)
    else:
        candidate_paths.append(Path.cwd() / path_obj)
        try:
            script_dir = Path(__file__).resolve().parent
        except NameError:
            script_dir = Path.cwd()
        candidate_paths.append(script_dir / path_obj)
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None

def load_env_file(path=".env"):
    try:
        if hasattr(st, "secrets"):
            for key, value in st.secrets.items():
                if key and (key not in os.environ or not os.environ.get(key)):
                    os.environ[key] = str(value)
    except Exception:
        pass
    env_path = resolve_env_path(path)
    if not env_path:
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value

def env_first(*keys, default=None):
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return default

def get_neo4j_credentials():
    load_env_file(".env")
    uri = env_first("NEO4J_URI", "NEO4J_BOLT_URL", "NEO4J_URL")
    user = env_first("NEO4J_USER", "NEO4J_USERNAME")
    password = env_first("NEO4J_PASSWORD", "NEO4J_PASS")
    database = env_first("NEO4J_DATABASE", "NEO4J_DB", default="neo4j")
    if not uri or not user or not password:
        missing = [k for k, v in {
            "NEO4J_URI/NEO4J_BOLT_URL": uri,
            "NEO4J_USER/NEO4J_USERNAME": user,
            "NEO4J_PASSWORD/NEO4J_PASS": password,
        }.items() if not v]
        raise RuntimeError("Missing Neo4j credentials: " + ", ".join(missing))
    return uri, user, password, database

def build_dbsm_rows_from_df(df: pd.DataFrame):
    required = {"File Name", "System Number", "Function Code", "Loop Sequence", "Tagname"}
    if not required.issubset(set(df.columns)):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV is missing required columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        tagname = str(row.get("Tagname", "")).strip()
        if not tagname:
            continue
        rows.append({
            "drawing_name": str(row.get("File Name", "")).strip(),
            "system_code": str(row.get("System Number", "")).strip(),
            "function_code": str(row.get("Function Code", "")).strip(),
            "loop_sequence": str(row.get("Loop Sequence", "")).strip(),
            "tagname": tagname
        })
    return rows

def build_dbsm_rows_from_extracted(extracted_data):
    if not extracted_data:
        return []
    df = pd.DataFrame(extracted_data)
    if "File Name" not in df.columns:
        df["File Name"] = ""
    df["System Number"] = df.get("System Number", "")
    df["Function Code"] = df.get("Function Code", "")
    df["Loop Sequence"] = df.get("Loop Sequence", "")
    df["Tagname"] = df.get("Tagname", "")
    return build_dbsm_rows_from_df(df[["File Name", "System Number", "Function Code", "Loop Sequence", "Tagname"]])

def build_unique_dbsm_nodes(rows):
    drawings = {}
    systems = {}
    functions = {}
    instruments = {}
    for row in rows:
        if row["drawing_name"]:
            drawings[row["drawing_name"]] = {"name": row["drawing_name"]}
        if row["system_code"]:
            systems[row["system_code"]] = {"code": row["system_code"]}
        if row["function_code"]:
            functions[row["function_code"]] = {"code": row["function_code"]}
        instruments[row["tagname"]] = {
            "tagname": row["tagname"],
            "loop_sequence": row["loop_sequence"],
            "system_number": row["system_code"],
            "function_code": row["function_code"],
        }
    return list(drawings.values()), list(systems.values()), list(functions.values()), list(instruments.values())

def build_unique_dbsm_relationships(rows):
    appears_in = set()
    belongs_to = set()
    has_function = set()
    for row in rows:
        if row["drawing_name"]:
            appears_in.add((row["tagname"], row["drawing_name"]))
        if row["system_code"]:
            belongs_to.add((row["tagname"], row["system_code"]))
        if row["function_code"]:
            has_function.add((row["tagname"], row["function_code"]))
    return (
        [{"instrument_tagname": t, "drawing_name": d} for t, d in appears_in],
        [{"instrument_tagname": t, "system_code": s} for t, s in belongs_to],
        [{"instrument_tagname": t, "function_code": f} for t, f in has_function],
    )

def chunked(items, size):
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]

def run_dbsm_population(rows, batch_size=500):
    constraints = [
        "CREATE CONSTRAINT drawing_name IF NOT EXISTS FOR (d:Drawing) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT system_code IF NOT EXISTS FOR (s:System) REQUIRE s.code IS UNIQUE",
        "CREATE CONSTRAINT function_code IF NOT EXISTS FOR (f:Function) REQUIRE f.code IS UNIQUE",
        "CREATE CONSTRAINT instrument_tag IF NOT EXISTS FOR (i:Instrument) REQUIRE i.tagname IS UNIQUE",
    ]

    upsert_drawings = """
    UNWIND $rows AS row
    MERGE (d:Drawing {name: row.name})
    SET d += row
    """
    upsert_systems = """
    UNWIND $rows AS row
    MERGE (s:System {code: row.code})
    SET s += row
    """
    upsert_functions = """
    UNWIND $rows AS row
    MERGE (f:Function {code: row.code})
    SET f += row
    """
    upsert_instruments = """
    UNWIND $rows AS row
    MERGE (i:Instrument {tagname: row.tagname})
    SET i.loop_sequence = row.loop_sequence,
        i.system_number = row.system_number,
        i.function_code = row.function_code
    """
    q_rel_appears_in = """
    UNWIND $rows AS row
    MATCH (i:Instrument {tagname: row.instrument_tagname})
    MATCH (d:Drawing {name: row.drawing_name})
    MERGE (i)-[:APPEARS_IN]->(d)
    """
    q_rel_belongs_to = """
    UNWIND $rows AS row
    MATCH (i:Instrument {tagname: row.instrument_tagname})
    MATCH (s:System {code: row.system_code})
    MERGE (i)-[:BELONGS_TO]->(s)
    """
    q_rel_has_function = """
    UNWIND $rows AS row
    MATCH (i:Instrument {tagname: row.instrument_tagname})
    MATCH (f:Function {code: row.function_code})
    MERGE (i)-[:HAS_FUNCTION]->(f)
    """

    uri, user, password, database = get_neo4j_credentials()
    drawings, systems, functions, instruments = build_unique_dbsm_nodes(rows)
    rel_appears_in, rel_belongs_to, rel_has_function = build_unique_dbsm_relationships(rows)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        for stmt in constraints:
            session.run(stmt)
        for batch in chunked(drawings, batch_size):
            session.run(upsert_drawings, rows=batch)
        for batch in chunked(systems, batch_size):
            session.run(upsert_systems, rows=batch)
        for batch in chunked(functions, batch_size):
            session.run(upsert_functions, rows=batch)
        for batch in chunked(instruments, batch_size):
            session.run(upsert_instruments, rows=batch)
        for batch in chunked(rel_appears_in, batch_size):
            session.run(q_rel_appears_in, rows=batch)
        for batch in chunked(rel_belongs_to, batch_size):
            session.run(q_rel_belongs_to, rows=batch)
        for batch in chunked(rel_has_function, batch_size):
            session.run(q_rel_has_function, rows=batch)

    driver.close()
    return {
        "drawings": len(drawings),
        "systems": len(systems),
        "functions": len(functions),
        "instruments": len(instruments),
        "rel_appears_in": len(rel_appears_in),
        "rel_belongs_to": len(rel_belongs_to),
        "rel_has_function": len(rel_has_function),
    }

def build_asset_table(extracted_data):
    """Build a unique asset table from extracted tag data."""
    if not extracted_data:
        return pd.DataFrame(columns=["Tagname", "System Number", "Function Code", "Loop Sequence", "lat", "lon"])
    df = pd.DataFrame(extracted_data)
    keep_cols = ["Tagname", "System Number", "Function Code", "Loop Sequence"]
    assets = df[keep_cols].drop_duplicates().reset_index(drop=True)
    assets["lat"] = np.nan
    assets["lon"] = np.nan
    return assets

def generate_cypher_script(extracted_data, asset_locations_df):
    """Generate a Neo4j Cypher script for a minimal KG demo."""
    if not extracted_data:
        return ""

    asset_lookup = {}
    if asset_locations_df is not None and not asset_locations_df.empty:
        for _, row in asset_locations_df.iterrows():
            tag = str(row.get("Tagname", "")).strip()
            lat_val = None if pd.isna(row.get("lat")) else float(row.get("lat"))
            lon_val = None if pd.isna(row.get("lon")) else float(row.get("lon"))
            asset_lookup[tag] = {"lat": lat_val, "lon": lon_val}

    rows = []
    for row in extracted_data:
        tag = row.get("Tagname")
        loc = asset_lookup.get(tag, {})
        rows.append({
            "diagram": row.get("File Name"),
            "tagname": tag,
            "system_number": row.get("System Number"),
            "function_code": row.get("Function Code"),
            "loop_sequence": row.get("Loop Sequence"),
            "symbol_class": row.get("Detection Class"),
            "symbol_confidence": row.get("Detection Confidence"),
            "symbol_bbox": row.get("BBox"),
            "ocr_text": row.get("OCR Text"),
            "lat": loc.get("lat"),
            "lon": loc.get("lon")
        })

    payload = json.dumps(rows, ensure_ascii=True)
    script = (
        "// Minimal KG import script (demo only)\n"
        "WITH " + payload + " AS rows\n"
        "UNWIND rows AS row\n"
        "MERGE (d:Diagram {name: row.diagram})\n"
        "MERGE (a:Asset {tagname: row.tagname})\n"
        "SET a.system_number = row.system_number,\n"
        "    a.function_code = row.function_code,\n"
        "    a.loop_sequence = row.loop_sequence\n"
        "FOREACH (_ IN CASE WHEN row.lat IS NULL OR row.lon IS NULL THEN [] ELSE [1] END |\n"
        "    SET a.lat = row.lat, a.lon = row.lon\n"
        ")\n"
        "CREATE (s:Symbol {\n"
        "    class: row.symbol_class,\n"
        "    confidence: row.symbol_confidence,\n"
        "    bbox: row.symbol_bbox,\n"
        "    ocr_text: row.ocr_text\n"
        "})\n"
        "MERGE (d)-[:HAS_SYMBOL]->(s)\n"
        "MERGE (s)-[:LABELLED_AS]->(a);\n"
    )
    return script

# -----------------------------------------------------------------------------
# NAVIGATION HELPERS FOR DISPLAYING IMAGES
# -----------------------------------------------------------------------------
def navigate_images(direction: str):
    """Navigate to previous or next processed image."""
    if direction == "prev":
        st.session_state["current_image_index"] = max(
            0, st.session_state["current_image_index"] - 1
        )
    elif direction == "next":
        total = len(st.session_state["processed_images"])
        st.session_state["current_image_index"] = min(
            total - 1, st.session_state["current_image_index"] + 1
        )

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
if not IS_EMBEDDED:
    st.set_page_config(
        layout="wide",
        page_title="P&ID Object Detection & Tag Extraction",
        page_icon=":mag:"
    )

st.title("P&ID Object Detection & Tag Extraction (Grid-based)")

# -----------------------------------------------------------------------------
# STEP 1: Model Selection & Loading
# -----------------------------------------------------------------------------
st.subheader("**Step 1:** Select & Load YOLO Model")

model_dir = "yolov5/runs/train"
available_models = [str(p) for p in Path(model_dir).glob("**/*.pt")]

if not available_models:
    st.warning("No YOLO .pt files found in the specified directory.")
else:
    selected_model = st.selectbox(
        "Choose a YOLO model file",
        available_models,
        help="Select your trained YOLO model."
    )
    load_model_btn = st.button("Load Model")

    if load_model_btn and selected_model:
        with st.spinner("Loading YOLO model..."):
            try:
                st.session_state["model"] = load_yolo_model(
                    selected_model,
                    device=DEVICE
                )
                st.success(f"Model loaded: {selected_model}")
            except Exception as ex:
                st.error(f"Failed to load the model: {ex}")

# -----------------------------------------------------------------------------
# STEP 2: File Upload
# -----------------------------------------------------------------------------
if st.session_state["model"] is not None:
    st.subheader("**Step 2:** Upload Your Files")

    st.markdown("Upload one or more P&ID files (PDF, JPG, PNG). If PDF, select your preferred DPI below.")

    pdf_dpi = st.number_input(
        "Select DPI for PDF Rendering (if uploading PDFs)",
        min_value=72,
        max_value=600,
        value=300,
        step=24,
        help="Higher DPI -> better resolution but larger memory usage."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF or image files",
        type=["pdf", "png", "jpg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Process the uploaded files
        all_images_local = []
        all_image_names_local = []

        proc = psutil.Process(os.getpid())
        pdf_before_mem = proc.memory_info().rss / (1024 * 1024)
        pdf_before_time = time.time()

        for file_obj in uploaded_files:
            fname = file_obj.name
            file_bytes = file_obj.read()
            file_obj.seek(0)

            if file_obj.type == "application/pdf":
                with st.spinner(f"Converting PDF '{fname}' to images..."):
                    pdf_imgs = render_pdf_to_images(
                        file_bytes,
                        dpi=pdf_dpi,   # use the DPI selected above
                        max_pages=50
                    )
                    for i, img in enumerate(pdf_imgs):
                        ds_img = maybe_downscale(img, max_dim=3000)
                        all_images_local.append(ds_img)
                        all_image_names_local.append(f"{fname} [Page {i+1}]")
            else:
                # It's an image
                try:
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
                    ds_img = maybe_downscale(pil_img, max_dim=3000)
                    all_images_local.append(ds_img)
                    all_image_names_local.append(fname)
                except Exception as e:
                    st.error(f"Error processing image file {fname}: {e}")

        pdf_after_mem = proc.memory_info().rss / (1024 * 1024)
        pdf_after_time = time.time()

        st.session_state["res_usage"]["pdf_mem_diff"] += (pdf_after_mem - pdf_before_mem)
        st.session_state["res_usage"]["pdf_time"] += (pdf_after_time - pdf_before_time)

        # Store in session state
        st.session_state["all_images"] = all_images_local
        st.session_state["all_image_names"] = all_image_names_local

        if len(st.session_state["all_images"]) > 0:
            st.success(f"Successfully uploaded {len(st.session_state['all_images'])} page(s)/image(s).")
        else:
            st.warning("No valid images were processed.")

# -----------------------------------------------------------------------------
# STEP 3: Detection Configuration
# -----------------------------------------------------------------------------
if st.session_state["model"] is not None and len(st.session_state["all_images"]) > 0:
    st.subheader("**Step 3:** Configure Detection Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["detection_config"]["conf_threshold"] = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["detection_config"]["conf_threshold"],
            step=0.05
        )
        st.session_state["detection_config"]["bbox_thickness"] = st.slider(
            "Bounding Box Thickness",
            1, 10,
            st.session_state["detection_config"]["bbox_thickness"]
        )
    with col2:
        st.session_state["detection_config"]["bbox_color"] = st.color_picker(
            "Bounding Box Color",
            st.session_state["detection_config"]["bbox_color"]
        )
        st.session_state["detection_config"]["color_by_class"] = st.checkbox(
            "Color by Class (override single color)",
            value=st.session_state["detection_config"]["color_by_class"]
        )

    with st.expander("Advanced Detection Settings (Optional)"):
        col3, col4 = st.columns(2)
        with col3:
            st.session_state["detection_config"]["nms_threshold"] = st.slider(
                "NMS IoU Threshold",
                0.0,
                1.0,
                st.session_state["detection_config"]["nms_threshold"],
                0.05
            )
            st.session_state["detection_config"]["apply_nms"] = st.checkbox(
                "Apply NMS?",
                value=st.session_state["detection_config"]["apply_nms"]
            )
        with col4:
            st.session_state["detection_config"]["dynamic_grid"] = st.checkbox(
                "Dynamic Grid Splitting?",
                value=st.session_state["detection_config"]["dynamic_grid"]
            )
            st.session_state["detection_config"]["max_grid_size"] = st.number_input(
                "Max Grid Size",
                min_value=1,
                max_value=16,
                value=st.session_state["detection_config"]["max_grid_size"]
            )

        col5, col6 = st.columns(2)
        with col5:
            st.session_state["detection_config"]["grid_size"] = (
                st.number_input("Grid Columns", 1, 16, st.session_state["detection_config"]["grid_size"][0]),
                st.number_input("Grid Rows", 1, 16, st.session_state["detection_config"]["grid_size"][1])
            )
        with col6:
            st.session_state["detection_config"]["inference_batch_size"] = st.number_input(
                "Inference Batch Size",
                1, 32,
                st.session_state["detection_config"]["inference_batch_size"]
            )
            st.session_state["detection_config"]["inference_timeout"] = st.number_input(
                "Inference Timeout (seconds)",
                1, 300,
                st.session_state["detection_config"]["inference_timeout"]
            )

    st.write("---")
    if st.button("Run Object Detection"):
        # Reset or prepare the processed_images list
        st.session_state["processed_images"] = []
        st.session_state["current_image_index"] = 0
        st.session_state["raw_detections"] = []

        detection_config = st.session_state["detection_config"]
        proc = psutil.Process(os.getpid())
        det_before_mem = proc.memory_info().rss / (1024 * 1024)
        det_before_time = time.time()

        for idx, img in enumerate(st.session_state["all_images"]):
            st.info(f"Detecting on image #{idx+1}: {st.session_state['all_image_names'][idx]}")
            detections = detect_objects_in_grids(
                model=st.session_state["model"],
                image=img,
                conf_threshold=detection_config["conf_threshold"],
                grid_size=detection_config["grid_size"],
                nms_threshold=detection_config["nms_threshold"],
                apply_nms=detection_config["apply_nms"],
                inference_batch_size=detection_config["inference_batch_size"],
                dynamic_grid=detection_config["dynamic_grid"],
                max_grid_size=detection_config["max_grid_size"]
            )

            st.session_state["raw_detections"].append(detections)

            # Draw bounding boxes on a copy of the image
            drawn_img = img.copy()
            draw = ImageDraw.Draw(drawn_img)
            try:
                font = ImageFont.truetype(FONT_PATH, 20) if FONT_PATH else ImageFont.load_default()
            except IOError:
                font = ImageFont.load_default()

            # Filter out low confidence
            filtered_det = [d for d in detections if d["confidence"] >= detection_config["conf_threshold"]]
            for obj in filtered_det:
                xmin, ymin, xmax, ymax = map(int, obj["bbox"])
                bbox_color = detection_config["bbox_color"]
                if detection_config.get("color_by_class"):
                    bbox_color = color_for_class(obj.get("class"))
                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    outline=bbox_color,
                    width=detection_config["bbox_thickness"]
                )
                label = f"{obj['class']} ({obj['confidence']:.2f})"
                text_size = draw.textsize(label, font=font)
                text_bg = Image.new('RGBA', text_size, bbox_color)
                drawn_img.paste(text_bg, (xmin, ymin), text_bg)
                draw.text((xmin, ymin), label, fill="#FFFFFF", font=font)

            # Store processed image in session state
            st.session_state["processed_images"].append(drawn_img)

        det_after_mem = proc.memory_info().rss / (1024 * 1024)
        det_after_time = time.time()

        st.session_state["res_usage"]["detect_mem_diff"] += (det_after_mem - det_before_mem)
        st.session_state["res_usage"]["detect_time"] += (det_after_time - det_before_time)

        st.success(f"Detection completed for {len(st.session_state['all_images'])} images.")

        # --- Display total number of detections ---
        total_detections = sum(len(dets) for dets in st.session_state["raw_detections"])
        st.info(f"**Total number of detections across all images: {total_detections}**")

# -----------------------------------------------------------------------------
# STEP 4: Display Detected Images with Navigation
# -----------------------------------------------------------------------------
if st.session_state.get("processed_images"):
    st.subheader("**Step 4:** Review Detected Objects (Navigate)")

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⬅️ Previous"):
            navigate_images("prev")
    with col3:
        if st.button("➡️ Next"):
            navigate_images("next")

    # Display current image
    idx = st.session_state["current_image_index"]
    total_proc = len(st.session_state["processed_images"])
    current_image = st.session_state["processed_images"][idx]
    st.image(
        current_image,
        caption=f"Image {idx + 1} of {total_proc}: {st.session_state['all_image_names'][idx]}",
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# STEP 5: (Optional) Image Enhancement
# -----------------------------------------------------------------------------
if st.session_state.get("raw_detections") is not None:
    detection_config = st.session_state["detection_config"]

    # Re-filter to ensure we have current threshold
    filtered_detections = []
    for raw_det in st.session_state["raw_detections"]:
        flt = [x for x in raw_det if x["confidence"] >= detection_config["conf_threshold"]]
        filtered_detections.append(flt)

    total_objects = sum(len(x) for x in filtered_detections)
    if total_objects > 0:
        with st.expander("**Step 5 (Optional): Image Enhancement**"):
            st.markdown("Enhance bounding boxes (cropped) before running OCR.")

            all_bboxes = []
            for img_idx, det_list in enumerate(filtered_detections):
                for det in det_list:
                    all_bboxes.append({
                        "img_idx": img_idx,
                        "bbox_idx": len(all_bboxes) + 1,
                        "bbox": det["bbox"],
                        "class": det["class"],
                        "confidence": det["confidence"]
                    })

            if all_bboxes:
                bbox_options = [
                    f"Image {b['img_idx']+1} ({st.session_state['all_image_names'][b['img_idx']]}) - "
                    f"BBox {b['bbox_idx']} ({b['class']}, {b['confidence']:.2f})"
                    for b in all_bboxes
                ]
                selected_label = st.selectbox("Select a BBox", bbox_options)

                if selected_label:
                    idx_selected = bbox_options.index(selected_label)
                    bbox_info = all_bboxes[idx_selected]
                    img_idx = bbox_info["img_idx"]
                    bbox = bbox_info["bbox"]

                    selected_img = st.session_state["all_images"][img_idx]
                    cropped_img = selected_img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

                    st.markdown(f"**Selected BBox**: {selected_label}")
                    st.image(cropped_img, caption="Original Cropped Region")

                    col1, col2 = st.columns(2)
                    with col1:
                        denoise_strength = st.slider("Denoise Strength", 0, 100, 10)
                        denoise_template_window_size = st.slider("Template Window Size", 3, 41, 7, step=2)
                        denoise_search_window = st.slider("Search Window Size", 3, 41, 21, step=2)
                    with col2:
                        thresholding = st.checkbox("Threshold (B&W)?", False)
                        deskew_angle = st.slider("Deskew Angle", -90, 90, 0)

                    if st.button("Apply Enhancement to This BBox"):
                        with st.spinner("Enhancing..."):
                            try:
                                enhanced_crop = enhance_images(
                                    [cropped_img],
                                    resize_factor=1.0,
                                    denoise_strength=denoise_strength,
                                    denoise_template_window_size=denoise_template_window_size,
                                    denoise_search_window=denoise_search_window,
                                    thresholding=thresholding,
                                    deskew_angle=deskew_angle
                                )[0]
                                st.image(enhanced_crop, caption="Enhanced Cropped Region")

                                # Paste enhanced image back
                                orig_w = int(bbox[2] - bbox[0])
                                orig_h = int(bbox[3] - bbox[1])
                                if enhanced_crop.size != (orig_w, orig_h):
                                    enhanced_crop = enhanced_crop.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

                                st.session_state["all_images"][img_idx].paste(
                                    enhanced_crop,
                                    (int(bbox[0]), int(bbox[1]))
                                )
                                st.success("Enhanced region pasted back into the original image.")

                            except Exception as e:
                                st.error(f"Enhancement error: {e}")

# -----------------------------------------------------------------------------
# STEP 6: OCR & Tag Extraction
# -----------------------------------------------------------------------------
if st.session_state.get("raw_detections") is not None:
    st.subheader("**Step 6:** OCR & Tag Extraction")

    # Prepare
    ocr_engine = get_ocr_engine()

    detection_config = st.session_state["detection_config"]
    filtered_detections = []
    for raw_det in st.session_state["raw_detections"]:
        flt = [x for x in raw_det if x["confidence"] >= detection_config["conf_threshold"]]
        filtered_detections.append(flt)

    total_boxes = sum(len(x) for x in filtered_detections)
    st.markdown(f"Total Boxes for OCR: **{total_boxes}**")

    if total_boxes > 0:
        # OCR Config
        custom_regex_use = st.checkbox("Use Custom Regex for Tag Extraction?", value=False)
        if custom_regex_use:
            user_regex = st.text_input("Enter your custom regex:", DEFAULT_REGEX)
        else:
            user_regex = None

        # Tag builder
        parts_options = ["None", "System Number", "Function Code", "Loop Sequence"]
        separators_options = ["-", ""]

        st.markdown("**Tagname Builder**:")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            part1 = st.selectbox("Part 1", parts_options, index=1)
            if part1 == "System Number":
                system_hint = st.text_input("System Number Hint", "13")
            else:
                system_hint = "13"
        with col_b:
            sep1 = st.selectbox("Separator 1", separators_options, index=0)
        with col_c:
            part2 = st.selectbox("Part 2", parts_options, index=2)
        with col_d:
            sep2 = st.selectbox("Separator 2", separators_options, index=0)
        with col_e:
            part3 = st.selectbox("Part 3", parts_options, index=3)

        user_parts = [part1, part2, part3]
        user_seps = [sep1, sep2]
        naming_regex = generate_tagname_regex(custom_regex=user_regex)

        show_cropped_regions = st.checkbox("Show Cropped Regions During OCR?", value=False)

        if st.button("Run OCR & Extract Tagnames"):
            proc = psutil.Process(os.getpid())
            ocr_before_mem = proc.memory_info().rss / (1024 * 1024)
            ocr_before_time = time.time()

            extracted_data = []
            count_so_far = 0
            ocr_progress = st.progress(0)

            for img_idx, (img, dets) in enumerate(zip(st.session_state["all_images"], filtered_detections)):
                for obj in dets:
                    xmin, ymin, xmax, ymax = map(int, obj["bbox"])
                    cropped_region = img.crop((xmin, ymin, xmax, ymax))

                    if show_cropped_regions:
                        st.image(cropped_region, caption=f"{st.session_state['all_image_names'][img_idx]}", width=300)

                    text = extract_text_with_ocr(ocr_engine, cropped_region)
                    normalized = normalize_text(text)
                    all_tag_parts = extract_tagnames_from_text(
                        normalized,
                        naming_regex,
                        system_hint=system_hint
                    )

                    for raw_tag in all_tag_parts:
                        sys_num = raw_tag["system_number"]
                        func_code = raw_tag["function_code"]
                        loop_seq = raw_tag["loop_sequence"]

                        final_tag = build_tagname(
                            system_number=sys_num,
                            function_code=func_code,
                            loop_sequence=loop_seq,
                            parts=user_parts,
                            separators=user_seps
                        )
                        extracted_data.append({
                            "File Name": st.session_state["all_image_names"][img_idx],
                            "System Number": sys_num,
                            "Function Code": func_code,
                            "Loop Sequence": loop_seq,
                            "Tagname": final_tag,
                            "Detection Class": obj["class"],
                            "Detection Confidence": obj["confidence"],
                            "BBox": f"{xmin},{ymin},{xmax},{ymax}",
                            "OCR Text": text
                        })

                    count_so_far += 1
                    ocr_progress.progress(int(count_so_far / total_boxes * 100))

            # -- AFTER BUILDING extracted_data --
            st.session_state["extracted_data"] = extracted_data
            assets_df = build_asset_table(extracted_data)
            if st.session_state["asset_locations"] is not None:
                prev_assets = st.session_state["asset_locations"][["Tagname", "lat", "lon"]]
                assets_df = assets_df.merge(prev_assets, on="Tagname", how="left", suffixes=("", "_prev"))
                assets_df["lat"] = assets_df["lat_prev"].combine_first(assets_df["lat"])
                assets_df["lon"] = assets_df["lon_prev"].combine_first(assets_df["lon"])
                assets_df = assets_df.drop(columns=["lat_prev", "lon_prev"])
            st.session_state["asset_locations"] = assets_df

            total_extracted_tagnames = len(extracted_data)
            st.info(f"**Total number of extracted tagnames: {total_extracted_tagnames}**")

            ocr_after_mem = proc.memory_info().rss / (1024 * 1024)
            ocr_after_time = time.time()

            st.session_state["res_usage"]["ocr_mem_diff"] += (ocr_after_mem - ocr_before_mem)
            st.session_state["res_usage"]["ocr_time"] += (ocr_after_time - ocr_before_time)

            if extracted_data:
                df = pd.DataFrame(extracted_data)
                st.dataframe(df, use_container_width=True)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV of Tagnames",
                    data=csv_bytes,
                    file_name="tagnames.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No tagnames recognized from OCR.")

# -----------------------------------------------------------------------------
# STEP 7: Knowledge Graph & Asset Map (Prototype)
# -----------------------------------------------------------------------------
if st.session_state.get("extracted_data"):
    st.subheader("**Step 7:** Knowledge Graph & Asset Map (Prototype)")
    st.markdown(
        "Minimal KG schema (v0.1): **Diagram → Symbol → Asset**. "
        "This is a prototype export to validate structure and queries."
    )

    extracted_df = pd.DataFrame(st.session_state["extracted_data"])
    diagram_count = extracted_df["File Name"].nunique()
    symbol_count = len(extracted_df)
    asset_count = extracted_df["Tagname"].nunique()
    st.info(f"Diagrams: **{diagram_count}** | Symbols: **{symbol_count}** | Assets: **{asset_count}**")

    if st.session_state["asset_locations"] is None or st.session_state["asset_locations"].empty:
        st.session_state["asset_locations"] = build_asset_table(st.session_state["extracted_data"])

    st.markdown("**Asset location entry (manual, demo only)**")
    edited_assets = st.data_editor(
        st.session_state["asset_locations"],
        column_config={
            "lat": st.column_config.NumberColumn("lat", help="Latitude"),
            "lon": st.column_config.NumberColumn("lon", help="Longitude")
        },
        use_container_width=True,
        num_rows="dynamic"
    )
    st.session_state["asset_locations"] = edited_assets

    use_placeholder = st.checkbox(
        "Use placeholder coordinates for missing assets (demo only)",
        value=False
    )
    plot_df = edited_assets.copy()
    plot_df["lat"] = pd.to_numeric(plot_df["lat"], errors="coerce")
    plot_df["lon"] = pd.to_numeric(plot_df["lon"], errors="coerce")
    if use_placeholder:
        plot_df["lat"] = plot_df["lat"].fillna(43.5)
        plot_df["lon"] = plot_df["lon"].fillna(33.8)
        st.caption("Placeholder coordinates applied (not real asset locations).")

    plot_df = plot_df.dropna(subset=["lat", "lon"])
    if not plot_df.empty:
        fig = px.scatter_geo(
            plot_df,
            lat="lat",
            lon="lon",
            hover_name="Tagname",
            scope="europe"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add lat/lon values to render the asset map.")

    cypher_script = generate_cypher_script(
        st.session_state["extracted_data"],
        st.session_state["asset_locations"]
    )
    if cypher_script:
        st.download_button(
            "Download Neo4j Cypher Script",
            data=cypher_script,
            file_name="blackseaguard_kg.cypher",
            mime="text/plain"
        )

        assets_csv = st.session_state["asset_locations"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Asset Locations CSV",
            data=assets_csv,
            file_name="asset_locations.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# STEP 8: Populate DBSM (Neo4j Sandbox)
# -----------------------------------------------------------------------------
if st.session_state.get("extracted_data"):
    st.subheader("**Step 8:** Populate DBSM (Neo4j Sandbox)")
    st.markdown(
        "Uses Neo4j credentials from `.env` (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD). "
        "This populates `Drawing`, `System`, `Function`, `Instrument` with edges."
    )

    data_source = st.radio(
        "Data source",
        ["Use current extracted data", "Use local CSV path"],
        horizontal=True
    )

    csv_path = None
    if data_source == "Use local CSV path":
        csv_path = st.text_input("CSV path", value="tagnames.csv")
    else:
        if not st.session_state.get("extracted_data"):
            st.warning("No extracted data in session. Run OCR or use a CSV path.")

    batch_size = st.number_input("Batch size", min_value=50, max_value=5000, value=500, step=50)

    if st.button("Populate DBSM"):
        try:
            if data_source == "Use current extracted data":
                rows = build_dbsm_rows_from_extracted(st.session_state.get("extracted_data"))
            else:
                if not csv_path or not os.path.exists(csv_path):
                    st.error("CSV path not found. Check the path and try again.")
                    rows = []
                else:
                    df = pd.read_csv(csv_path)
                    rows = build_dbsm_rows_from_df(df)

            if not rows:
                st.warning("No rows to load.")
            else:
                with st.spinner("Populating Neo4j..."):
                    stats = run_dbsm_population(rows, batch_size=int(batch_size))
                st.success(
                    "Loaded "
                    f"{stats['instruments']} instruments, "
                    f"{stats['drawings']} drawings, "
                    f"{stats['systems']} systems, "
                    f"{stats['functions']} functions."
                )
        except Exception as exc:
            st.error(f"Population failed: {exc}")

# -----------------------------------------------------------------------------
