import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PIL import Image


# -----------------------------
# Streamlit compatibility
# -----------------------------

def st_dataframe_compat(df, **kwargs):
    try:
        return st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)


# -----------------------------
# Filename parsing
# -----------------------------

TIME_OF_YEAR_MAP = {
    "boy": "Beginning",
    "moy": "Mid",
    "eoy": "End",
}

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Expected:
      [school_number] gr [grade number] [time of year] [assessment_name].pdf

    Example:
      194 gr 8 boy map.pdf
    """
    stem = Path(filename).stem.strip()

    pattern = re.compile(
        r"^(?P<school>\d+)\s+gr\s+(?P<grade>\d+)\s+(?P<toy>boy|moy|eoy)\s+(?P<assessment>.+)$",
        re.IGNORECASE,
    )
    m = pattern.match(stem)

    if not m:
        raise ValueError(
            f"Filename does not match expected pattern: '{filename}'. "
            "Expected format: '[school_number] gr [grade number] [boy/moy/eoy] [assessment_name].pdf'"
        )

    toy_code = m.group("toy").lower()

    return {
        "school": str(m.group("school")),
        "grade": str(m.group("grade")),
        "time_of_year": TIME_OF_YEAR_MAP.get(toy_code, toy_code),
        "assessment_name": str(m.group("assessment")).strip().upper(),
    }


# -----------------------------
# Color classification
# -----------------------------

BAND_COLORS = {
    "% 1st-20th": np.array([175, 0, 0]),       # dark red
    "% 21st-40th": np.array([255, 155, 90]),   # orange
    "% 41st-60th": np.array([255, 225, 90]),   # yellow
    "% 61st-80th": np.array([95, 195, 105]),   # green
    "% >80th": np.array([0, 95, 160]),         # dark blue
}

EMPTY_BANDS = {
    "% 1st-20th": 0,
    "% 21st-40th": 0,
    "% 41st-60th": 0,
    "% 61st-80th": 0,
    "% >80th": 0,
}

def classify_color(rgb: np.ndarray) -> Optional[str]:
    """
    Classify sampled bar background color into percentile band.
    """
    if rgb is None or np.isnan(rgb).any():
        return None

    distances = {
        band: np.linalg.norm(rgb.astype(float) - color.astype(float))
        for band, color in BAND_COLORS.items()
    }
    best_band = min(distances, key=distances.get)

    # Conservative threshold; fallback handled later if no band found.
    if distances[best_band] > 120:
        return None

    return best_band


def sample_background_color(
    page_img: Image.Image,
    word: Dict,
    zoom: float = 3.0,
    pad: int = 8,
) -> Optional[np.ndarray]:
    """
    Sample colored background behind a percent label.
    Ignores near-white and near-black pixels so text doesn't dominate.
    """
    arr = np.array(page_img.convert("RGB"))

    x0 = max(int(word["x0"] * zoom) - pad, 0)
    x1 = min(int(word["x1"] * zoom) + pad, arr.shape[1])
    y0 = max(int(word["top"] * zoom) - pad, 0)
    y1 = min(int(word["bottom"] * zoom) + pad, arr.shape[0])

    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    flat = patch.reshape(-1, 3)

    # Remove black/white/gray-ish text/background; keep saturated colored pixels.
    maxc = flat.max(axis=1)
    minc = flat.min(axis=1)
    saturation = maxc - minc

    mask = (
        (maxc < 245) &
        (minc > 20) &
        (saturation > 40)
    )

    colored = flat[mask]
    if len(colored) == 0:
        return None

    return np.median(colored, axis=0)


# -----------------------------
# PDF rendering and word helpers
# -----------------------------

def render_pdf_page(pdf_bytes: bytes, page_index: int, zoom: float = 3.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def word_text(w: Dict) -> str:
    return str(w.get("text", "")).strip()


def extract_class_code(text: str) -> Optional[str]:
    """
    Extract class code inside parentheses:
      (11X194.801.00)
      Art 8 (11X194.AQNM8.1)
    """
    m = re.search(r"\(([^)]+)\)", text)
    return m.group(1).strip() if m else None


def extract_median_percentile(text: str) -> Optional[int]:
    """
    Extract 50 from 50th, 41 from 41st, etc.
    """
    m = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)\b", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_percent_value(text: str) -> Optional[int]:
    """
    Extract 16 from '16%'.
    Also allows standalone small numbers like '4' used in the PDF's blue segment.
    """
    s = str(text).strip()

    m = re.fullmatch(r"(\d{1,3})%", s)
    if m:
        return int(m.group(1))

    # Some tiny blue segments appear as just "4" without the percent sign.
    m = re.fullmatch(r"(\d{1,3})", s)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val

    return None


def group_rows_from_page_words(words: List[Dict]) -> List[Tuple[float, float]]:
    """
    Detect table rows by locating class-code words in the left class column.
    Returns row vertical bounds as (y_top, y_bottom).
    """
    class_code_words = []

    for w in words:
        txt = word_text(w)
        x0 = w["x0"]

        # Left-side class column in this report layout.
        if x0 < 210 and re.search(r"\([^)]*\)", txt):
            class_code_words.append(w)

    if not class_code_words:
        return []

    class_code_words = sorted(class_code_words, key=lambda w: w["top"])
    ys = [w["top"] for w in class_code_words]

    row_bounds = []
    for i, y in enumerate(ys):
        if i == 0:
            y_top = max(y - 35, 0)
        else:
            y_top = (ys[i - 1] + y) / 2

        if i == len(ys) - 1:
            y_bottom = y + 45
        else:
            y_bottom = (y + ys[i + 1]) / 2

        row_bounds.append((y_top, y_bottom))

    return row_bounds


def words_in_region(words: List[Dict], x0: float, x1: float, y0: float, y1: float) -> List[Dict]:
    return [
        w for w in words
        if w["x0"] >= x0 and w["x1"] <= x1 and w["top"] >= y0 and w["bottom"] <= y1
    ]


def join_words(words: List[Dict]) -> str:
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    return " ".join(word_text(w) for w in words).strip()


# -----------------------------
# Main extraction
# -----------------------------

def extract_rows_from_pdf(pdf_bytes: bytes, filename: str) -> pd.DataFrame:
    meta = parse_filename(filename)
    records = []

    with pdfplumber.open(pd.io.common.BytesIO(pdf_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""

            # Skip pages before the relevant section appears unless they contain row continuations.
            # In the sample, the section starts on page 2.
            if (
                "Achievement by Class" not in page_text
                and "Class" not in page_text
                and "Achievement" not in page_text
            ):
                continue

            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )

            row_bounds = group_rows_from_page_words(words)
            if not row_bounds:
                continue

            page_img = render_pdf_page(pdf_bytes, page_index, zoom=3.0)

            for y0, y1 in row_bounds:
                class_words = words_in_region(words, 35, 205, y0, y1)
                educator_words = words_in_region(words, 205, 305, y0, y1)
                achievement_words = words_in_region(words, 305, 665, y0, y1)
                student_words = words_in_region(words, 665, 735, y0, y1)

                class_text = join_words(class_words)
                educator_text = join_words(educator_words)
                achievement_text = join_words(achievement_words)
                student_text = join_words(student_words)

                class_number = extract_class_code(class_text)
                if not class_number:
                    continue

                median_percentile = extract_median_percentile(achievement_text)

                num_students = None
                student_numbers = re.findall(r"\b\d+\b", student_text)
                if student_numbers:
                    num_students = int(student_numbers[-1])

                bands = EMPTY_BANDS.copy()

                # Find percent labels in the achievement area and classify by sampled background color.
                pct_words = []
                for w in achievement_words:
                    val = extract_percent_value(word_text(w))
                    if val is None:
                        continue

                    # Skip median percentile pill like 50th/41st; this helper only returns for plain 50 or 50%.
                    txt = word_text(w)
                    if re.search(r"(st|nd|rd|th)$", txt, flags=re.IGNORECASE):
                        continue

                    # Keep words in the distribution-bar area, not the median pill area.
                    if w["x0"] < 350:
                        continue

                    pct_words.append((w, val))

                for w, val in pct_words:
                    rgb = sample_background_color(page_img, w, zoom=3.0)
                    band = classify_color(rgb)
                    if band:
                        bands[band] = val

                records.append({
                    **meta,
                    "class_number": str(class_number),
                    "teacher": str(educator_text),
                    "median_achievement_percentile": median_percentile,
                    "% 1st-20th": bands["% 1st-20th"],
                    "% 21st-40th": bands["% 21st-40th"],
                    "% 41st-60th": bands["% 41st-60th"],
                    "% 61st-80th": bands["% 61st-80th"],
                    "% >80th": bands["% >80th"],
                    "number_of_students": num_students,
                    "source_file": filename,
                })

    return pd.DataFrame(records)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PDF Achievement by Class Extractor", layout="wide")
st.title("PDF Achievement by Class Extractor")

st.write(
    "Upload one or more MAP achievement PDFs. "
    "The app extracts school, grade, time of year, and assessment from the filename."
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

if st.button("Extract all PDFs", type="primary"):
    all_outputs = []
    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded_files, start=1):
        status.write(f"Extracting {i}/{len(uploaded_files)}: **{f.name}**")

        try:
            pdf_bytes = f.read()
            extracted = extract_rows_from_pdf(pdf_bytes, f.name)

            if extracted.empty:
                st.warning(f"No rows extracted from {f.name}.")
            else:
                all_outputs.append(extracted)

        except Exception as e:
            st.error(f"Extraction failed for {f.name}: {e}")

        progress.progress(i / len(uploaded_files))

    status.empty()

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)

        col_order = [
            "school",
            "grade",
            "time_of_year",
            "assessment_name",
            "class_number",
            "teacher",
            "median_achievement_percentile",
            "% 1st-20th",
            "% 21st-40th",
            "% 41st-60th",
            "% 61st-80th",
            "% >80th",
            "number_of_students",
            "source_file",
        ]
        combined = combined[[c for c in col_order if c in combined.columns]]

        st.success(f"Extracted {len(combined)} row(s).")
        st_dataframe_compat(combined)

        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="pdf_achievement_by_class_extract.csv",
            mime="text/csv",
        )
    else:
        st.error("No data was extracted from the uploaded PDFs.")
else:
    st.info("Click **Extract all PDFs** to run extraction.")
