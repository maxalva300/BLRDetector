import base64
import csv
import io
import math
import time
import zipfile

import cv2
import numpy as np
import plotly.graph_objects as go

from utils.image_processing import (
    _build_imagej_export_mask,
    _split_export_into_single_piece_regions,
    _remove_long_dark_strip_artifacts,
)


# ---------------------------------------------------------
# Metadata geometry
# ---------------------------------------------------------

def _normalize_column_name(column):
    """
    Make Metadata.xls column names comparable across batches/readers.
    """
    if isinstance(column, tuple):
        text = ".".join(
            str(part)
            for part in column
            if str(part).strip()
        )
    else:
        text = str(column)

    text = text.strip().lower()

    return "".join(
        ch
        for ch in text
        if ch.isalnum()
    )


def _find_column(df, aliases):
    aliases = {
        "".join(
            ch
            for ch in str(alias).lower()
            if ch.isalnum()
        )
        for alias in aliases
    }

    for column in df.columns:
        normalized = _normalize_column_name(
            column
        )

        if normalized in aliases:
            return column

    return None


def _safe_float(value):
    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def extract_metadata_geometry(df):
    """
    Read the batch-independent geometry used for QC:
      - width
      - height
      - pixel area

    Scanner coordinates are intentionally NOT used.
    """

    width_col = _find_column(
        df,
        {
            "size.width",
            "size_width",
            "width",
        },
    )

    height_col = _find_column(
        df,
        {
            "size.height",
            "size_height",
            "height",
        },
    )

    area_col = _find_column(
        df,
        {
            "area.pixel",
            "area_pixel",
            "pixel",
        },
    )

    if (
        width_col is None
        or height_col is None
        or area_col is None
    ):
        available = ", ".join(
            str(column)
            for column in df.columns
        )

        raise ValueError(
            "Metadata geometry columns could not be identified. "
            "Expected size.width, size.height and area.pixel. "
            f"Columns found: {available}"
        )

    records = []

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        width = _safe_float(
            row[width_col]
        )

        height = _safe_float(
            row[height_col]
        )

        area = _safe_float(
            row[area_col]
        )

        if (
            width is None
            or height is None
            or area is None
        ):
            continue

        if (
            width <= 0
            or height <= 0
            or area <= 0
        ):
            continue

        records.append(
            {
                "metadata_id":
                    row_number,
                "width":
                    width,
                "height":
                    height,
                "area":
                    area,
            }
        )

    if not records:
        raise ValueError(
            "No valid object geometry was found in Metadata.xls."
        )

    return records


# ---------------------------------------------------------
# Fast segmentation
# ---------------------------------------------------------

def _decode_image(image_contents):
    _, encoded = image_contents.split(
        ",",
        1,
    )

    image_bytes = base64.b64decode(
        encoded
    )

    np_buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    bgr = cv2.imdecode(
        np_buffer,
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise ValueError(
            "Could not decode RGB shelf image."
        )

    return bgr


def _prepare_image(bgr):
    """
    Compute expensive transforms once per batch.
    """

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV,
    )

    brightness = hsv[:, :, 2]

    channel_max = np.max(
        bgr,
        axis=2,
    )

    channel_min = np.min(
        bgr,
        axis=2,
    )

    channel_mean = np.mean(
        bgr,
        axis=2,
    )

    neutral_difference = (
        channel_max - channel_min
    )

    shelf_mask = (
        (neutral_difference <= 3)
        & (channel_mean >= 115)
        & (channel_mean <= 140)
    )

    return brightness, shelf_mask


def _segment_prepared(
    brightness,
    shelf_mask,
    brightness_min,
    kernel_size,
    min_area,
    max_area,
):
    foreground = (
        (brightness >= brightness_min)
        & (~shelf_mask)
    )

    binary = (
        foreground.astype(np.uint8)
        * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    (
        number_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    candidates = []

    for label in range(
        1,
        number_labels,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        if area > max_area:
            continue

        candidates.append(
            {
                "label":
                    label,
                "x":
                    int(
                        stats[
                            label,
                            cv2.CC_STAT_LEFT,
                        ]
                    ),
                "y":
                    int(
                        stats[
                            label,
                            cv2.CC_STAT_TOP,
                        ]
                    ),
                "width":
                    int(
                        stats[
                            label,
                            cv2.CC_STAT_WIDTH,
                        ]
                    ),
                "height":
                    int(
                        stats[
                            label,
                            cv2.CC_STAT_HEIGHT,
                        ]
                    ),
                "area":
                    area,
            }
        )

    return labels, candidates


# ---------------------------------------------------------
# Geometry compatibility
# ---------------------------------------------------------

def _metadata_arrays(
    metadata_records
):
    widths = np.asarray(
        [
            record["width"]
            for record in metadata_records
        ],
        dtype=float,
    )

    heights = np.asarray(
        [
            record["height"]
            for record in metadata_records
        ],
        dtype=float,
    )

    areas = np.asarray(
        [
            record["area"]
            for record in metadata_records
        ],
        dtype=float,
    )

    major = np.maximum(
        widths,
        heights,
    )

    minor = np.minimum(
        widths,
        heights,
    )

    return major, minor, areas


def geometry_compatibility_scores(
    candidates,
    metadata_records,
):
    """
    Measure how compatible each image candidate is with ANY object
    geometry present in this batch Metadata.

    This is a QC filter, not forced object-to-object matching.
    """

    if not candidates:
        return np.asarray(
            [],
            dtype=float,
        )

    (
        metadata_major,
        metadata_minor,
        metadata_area,
    ) = _metadata_arrays(
        metadata_records
    )

    candidate_major = np.asarray(
        [
            max(
                candidate["width"],
                candidate["height"],
            )
            for candidate in candidates
        ],
        dtype=float,
    )[:, None]

    candidate_minor = np.asarray(
        [
            min(
                candidate["width"],
                candidate["height"],
            )
            for candidate in candidates
        ],
        dtype=float,
    )[:, None]

    candidate_area = np.asarray(
        [
            candidate["area"]
            for candidate in candidates
        ],
        dtype=float,
    )[:, None]

    major_error = (
        np.abs(
            np.log(
                (candidate_major + 1.0)
                / (
                    metadata_major[
                        None,
                        :
                    ]
                    + 1.0
                )
            )
        )
        / 0.45
    )

    minor_error = (
        np.abs(
            np.log(
                (candidate_minor + 1.0)
                / (
                    metadata_minor[
                        None,
                        :
                    ]
                    + 1.0
                )
            )
        )
        / 0.45
    )

    area_error = (
        np.abs(
            np.log(
                (candidate_area + 1.0)
                / (
                    metadata_area[
                        None,
                        :
                    ]
                    + 1.0
                )
            )
        )
        / 0.90
    )

    pair_score = np.sqrt(
        major_error ** 2
        + minor_error ** 2
        + 0.50
        * area_error ** 2
    )

    return np.min(
        pair_score,
        axis=1,
    )


# ---------------------------------------------------------
# Geometry-aware automatic parameter selection
# ---------------------------------------------------------

def analyze_geometry_qc(
    image_contents,
    metadata_df,
    min_area=20,
    max_area=8000,
    geometry_threshold=1.0,
):
    """
    Fast automatic detector + Metadata geometry QC.

    V0.54 optimization:
      - expensive image transforms are computed once
      - the 40-setting search does NOT keep a full-size label matrix
        for the current best result
      - the chosen segmentation is rebuilt only once after the search

    No scanner-coordinate transformation.
    No giant multi-threshold candidate pool.
    No forced object-to-object assignment.
    """

    started = time.perf_counter()

    bgr = _decode_image(image_contents)
    brightness, shelf_mask = _prepare_image(bgr)

    metadata_records = extract_metadata_geometry(metadata_df)
    target_count = len(metadata_df)

    brightness_values = [8, 10, 12, 15, 18, 22, 26, 30]
    kernel_values = [3, 5, 7, 9, 11]

    best = None
    evaluated_settings = 0

    for brightness_min in brightness_values:
        for kernel_size in kernel_values:
            evaluated_settings += 1

            # IMPORTANT: trial labels are temporary. We deliberately do
            # not retain the full-size matrix in `best`, reducing peak RAM.
            trial_labels, candidates = _segment_prepared(
                brightness,
                shelf_mask,
                brightness_min,
                kernel_size,
                min_area,
                max_area,
            )

            geometry_scores = geometry_compatibility_scores(
                candidates,
                metadata_records,
            )

            accepted_mask = geometry_scores <= geometry_threshold
            accepted_count = int(np.count_nonzero(accepted_mask))
            suspicious_count = len(candidates) - accepted_count

            if accepted_count < target_count:
                count_penalty = (target_count - accepted_count) * 2.0
            else:
                count_penalty = (accepted_count - target_count) * 1.0

            selection_score = (
                count_penalty
                + 0.10 * suspicious_count
                + 0.02 * abs(brightness_min - 12)
                + 0.02 * abs(kernel_size - 5)
            )

            result = {
                "selection_score": float(selection_score),
                "brightness_min": int(brightness_min),
                "kernel_size": int(kernel_size),
                "accepted_count": int(accepted_count),
                "suspicious_count": int(suspicious_count),
            }

            if best is None or result["selection_score"] < best["selection_score"]:
                best = result

            # Make the lifetime explicit; on large shelves this matrix is
            # the largest temporary allocation in the parameter search.
            del trial_labels

    # Rebuild only the selected setting. This is the only label matrix that
    # survives into preview/export.
    labels, best_candidates = _segment_prepared(
        brightness,
        shelf_mask,
        best["brightness_min"],
        best["kernel_size"],
        min_area,
        max_area,
    )

    best_scores = geometry_compatibility_scores(
        best_candidates,
        metadata_records,
    )

    accepted = []
    suspicious = []

    for candidate, score in zip(best_candidates, best_scores):
        item = dict(candidate)
        item["geometry_score"] = float(score)

        if score <= geometry_threshold:
            accepted.append(item)
        else:
            suspicious.append(item)

    accepted.sort(key=lambda obj: (obj["y"], obj["x"]))
    suspicious.sort(key=lambda obj: (obj["y"], obj["x"]))

    elapsed = time.perf_counter() - started

    diagnostics = {
        "scanner_count": int(target_count),
        "metadata_geometry_count": int(len(metadata_records)),
        "raw_candidate_count": int(len(best_candidates)),
        "automatic_accepted_count": int(len(accepted)),
        "automatic_suspicious_count": int(len(suspicious)),
        "brightness_min": int(best["brightness_min"]),
        "kernel_size": int(best["kernel_size"]),
        "geometry_threshold": float(geometry_threshold),
        "evaluated_settings": int(evaluated_settings),
        "elapsed_seconds": float(elapsed),
    }

    return {
        "bgr": bgr,
        "labels": labels,
        "accepted": accepted,
        "suspicious": suspicious,
        "diagnostics": diagnostics,
    }


def prepare_geometry_qc_cache(
    image_contents,
    metadata_df,
    min_area=20,
    max_area=8000,
    geometry_threshold=1.0,
):
    """
    Run the expensive 40-setting detector search once and return only
    JSON-safe, compact information suitable for dcc.Store.

    Full images, label matrices and masks are deliberately NOT stored.
    """
    analysis = analyze_geometry_qc(
        image_contents,
        metadata_df,
        min_area=min_area,
        max_area=max_area,
        geometry_threshold=geometry_threshold,
    )

    score_by_label = {}
    for candidate in analysis["accepted"] + analysis["suspicious"]:
        score_by_label[str(int(candidate["label"]))] = float(
            candidate.get("geometry_score", 0.0)
        )

    return {
        "version": "0.54",
        "brightness_min": int(analysis["diagnostics"]["brightness_min"]),
        "kernel_size": int(analysis["diagnostics"]["kernel_size"]),
        "geometry_threshold": float(geometry_threshold),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "accepted_labels": [
            int(candidate["label"]) for candidate in analysis["accepted"]
        ],
        "suspicious_labels": [
            int(candidate["label"]) for candidate in analysis["suspicious"]
        ],
        "geometry_scores": score_by_label,
        "diagnostics": dict(analysis["diagnostics"]),
    }


def _analysis_from_geometry_qc_cache(
    image_contents,
    metadata_df,
    qc_cache,
):
    """
    Reconstruct one selected segmentation from the compact Analyze cache.
    This replaces the 40-setting search during Review and Generate.
    """
    if not qc_cache:
        return analyze_geometry_qc(image_contents, metadata_df)

    started = time.perf_counter()

    bgr = _decode_image(image_contents)
    brightness, shelf_mask = _prepare_image(bgr)

    labels, candidates = _segment_prepared(
        brightness,
        shelf_mask,
        int(qc_cache.get("brightness_min", 12)),
        int(qc_cache.get("kernel_size", 5)),
        int(qc_cache.get("min_area", 20)),
        int(qc_cache.get("max_area", 8000)),
    )

    accepted_labels = {
        int(value) for value in qc_cache.get("accepted_labels", [])
    }
    suspicious_labels = {
        int(value) for value in qc_cache.get("suspicious_labels", [])
    }
    score_by_label = qc_cache.get("geometry_scores", {}) or {}
    geometry_threshold = float(qc_cache.get("geometry_threshold", 1.0))

    accepted = []
    suspicious = []

    # Label IDs are deterministic for the same image + selected setting.
    # If a label is missing from cache for any reason, fall back to its
    # geometry score instead of silently discarding the candidate.
    metadata_records = None

    for candidate in candidates:
        item = dict(candidate)
        label = int(item["label"])
        cached_score = score_by_label.get(str(label))

        if cached_score is None:
            if metadata_records is None:
                metadata_records = extract_metadata_geometry(metadata_df)
            calculated = geometry_compatibility_scores([item], metadata_records)
            score = float(calculated[0]) if len(calculated) else 999.0
        else:
            score = float(cached_score)

        item["geometry_score"] = score

        if label in accepted_labels:
            accepted.append(item)
        elif label in suspicious_labels:
            suspicious.append(item)
        elif score <= geometry_threshold:
            accepted.append(item)
        else:
            suspicious.append(item)

    accepted.sort(key=lambda obj: (obj["y"], obj["x"]))
    suspicious.sort(key=lambda obj: (obj["y"], obj["x"]))

    diagnostics = dict(qc_cache.get("diagnostics", {}))
    diagnostics.update({
        "raw_candidate_count": int(len(candidates)),
        "automatic_accepted_count": int(len(accepted)),
        "automatic_suspicious_count": int(len(suspicious)),
        "cache_used": True,
        "rebuild_seconds": float(time.perf_counter() - started),
    })

    return {
        "bgr": bgr,
        "labels": labels,
        "accepted": accepted,
        "suspicious": suspicious,
        "diagnostics": diagnostics,
    }


def _apply_manual_review(
    analysis,
    manual_accepted_labels=None,
    manual_excluded_labels=None,
):
    """
    Apply optional human review while keeping only two visual states.

    GREEN  = included in download.
    YELLOW = excluded from download.

    manual_accepted_labels: yellow -> green
    manual_excluded_labels: green -> yellow

    If a label appears in both sets, exclusion wins.
    """

    manual_accept_set = {
        int(label)
        for label in (
            manual_accepted_labels
            or []
        )
    }

    manual_exclude_set = {
        int(label)
        for label in (
            manual_excluded_labels
            or []
        )
    }

    accepted = []
    suspicious = []

    promoted_count = 0
    excluded_count = 0

    # Automatic greens.
    for candidate in analysis["accepted"]:
        item = dict(candidate)
        label = int(item["label"])

        if label in manual_exclude_set:
            suspicious.append(item)
            excluded_count += 1
        else:
            accepted.append(item)

    # Automatic yellows.
    for candidate in analysis["suspicious"]:
        item = dict(candidate)
        label = int(item["label"])

        if (
            label in manual_accept_set
            and label not in manual_exclude_set
        ):
            accepted.append(item)
            promoted_count += 1
        else:
            suspicious.append(item)

    accepted.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

    suspicious.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

    for index, candidate in enumerate(
        accepted,
        start=1,
    ):
        candidate["object_number"] = index

    for index, candidate in enumerate(
        suspicious,
        start=1,
    ):
        candidate["suspicious_number"] = index

    diagnostics = dict(
        analysis["diagnostics"]
    )

    diagnostics["manual_accepted_count"] = promoted_count
    diagnostics["manual_excluded_count"] = excluded_count
    diagnostics["accepted_count"] = len(accepted)
    diagnostics["suspicious_count"] = len(suspicious)
    diagnostics["difference"] = (
        len(accepted)
        - diagnostics["scanner_count"]
    )
    diagnostics["ready"] = (
        len(accepted)
        == diagnostics["scanner_count"]
    )

    return {
        "bgr":
            analysis["bgr"],
        "labels":
            analysis["labels"],
        "accepted":
            accepted,
        "suspicious":
            suspicious,
        "diagnostics":
            diagnostics,
    }


# ---------------------------------------------------------
# Manual Add Object - Phase 1
# ---------------------------------------------------------

def _odd_kernel(value):
    value = int(
        max(
            3,
            value,
        )
    )

    if value % 2 == 0:
        value += 1

    return value


def _clip_manual_roi(
    roi,
    image_width,
    image_height,
):
    """
    Normalize and clip a Plotly rectangle to image coordinates.
    """

    x0 = int(
        round(
            min(
                float(
                    roi["x0"]
                ),
                float(
                    roi["x1"]
                ),
            )
        )
    )

    x1 = int(
        round(
            max(
                float(
                    roi["x0"]
                ),
                float(
                    roi["x1"]
                ),
            )
        )
    )

    y0 = int(
        round(
            min(
                float(
                    roi["y0"]
                ),
                float(
                    roi["y1"]
                ),
            )
        )
    )

    y1 = int(
        round(
            max(
                float(
                    roi["y0"]
                ),
                float(
                    roi["y1"]
                ),
            )
        )
    )

    x0 = max(
        0,
        min(
            image_width - 1,
            x0,
        ),
    )

    x1 = max(
        1,
        min(
            image_width,
            x1,
        ),
    )

    y0 = max(
        0,
        min(
            image_height - 1,
            y0,
        ),
    )

    y1 = max(
        1,
        min(
            image_height,
            y1,
        ),
    )

    if (
        x1 - x0 < 8
        or y1 - y0 < 8
    ):
        raise ValueError(
            "The Add Object box is too small. "
            "Draw a box around one complete object with a small margin."
        )

    return (
        x0,
        y0,
        x1,
        y1,
    )


def _manual_localization_mask(
    crop_bgr,
):
    """
    Build one permissive localization mask inside a user-selected ROI.

    This is deliberately used ONLY for manual repair/addition.
    The user has already constrained the search to one physical object.

    The final exported pixels still go through the validated
    ImageJ-compatible export path.
    """

    crop_h, crop_w = (
        crop_bgr.shape[:2]
    )

    hsv = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2HSV,
    )

    brightness = hsv[:, :, 2]

    # Remove the artificial grey shelf exactly as in the
    # automatic detector.
    channel_max = np.max(
        crop_bgr,
        axis=2,
    )

    channel_min = np.min(
        crop_bgr,
        axis=2,
    )

    channel_mean = np.mean(
        crop_bgr,
        axis=2,
    )

    shelf_mask = (
        (
            channel_max
            - channel_min
        )
        <= 3
    ) & (
        channel_mean
        >= 115
    ) & (
        channel_mean
        <= 140
    )

    min_dimension = max(
        1,
        min(
            crop_w,
            crop_h,
        ),
    )

    # A larger close is acceptable here because the user has
    # explicitly selected ONE object and this tool is meant to
    # reconnect fragmented detections.
    close_kernel_size = _odd_kernel(
        min(
            25,
            max(
                5,
                round(
                    min_dimension
                    * 0.12
                ),
            ),
        )
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                close_kernel_size,
                close_kernel_size,
            ),
        )
    )

    blurred = cv2.GaussianBlur(
        brightness,
        (
            5,
            5,
        ),
        0,
    )

    selected = None
    selected_threshold = None

    # Prefer the most permissive valid threshold.
    # If a very low threshold floods the ROI, automatically
    # move upward.
    for threshold in [
        5,
        8,
        10,
        12,
        15,
    ]:
        binary = (
            (
                blurred
                >= threshold
            )
            & (
                ~shelf_mask
            )
        ).astype(
            np.uint8
        ) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        (
            component_count,
            component_labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        if component_count <= 1:
            continue

        # Keep the largest local component. The large close above
        # is what reconnects pieces of a fragmented object.
        component_areas = (
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )

        largest_component = (
            1
            + int(
                np.argmax(
                    component_areas
                )
            )
        )

        candidate_mask = (
            component_labels
            == largest_component
        ).astype(
            np.uint8
        ) * 255

        candidate_area = int(
            np.count_nonzero(
                candidate_mask
            )
        )

        roi_area = (
            crop_w
            * crop_h
        )

        area_ratio = (
            candidate_area
            / max(
                1,
                roi_area,
            )
        )

        # Guard against background flooding while still allowing
        # a tightly drawn box.
        if (
            area_ratio < 0.005
            or area_ratio > 0.85
        ):
            continue

        selected = candidate_mask
        selected_threshold = threshold
        break

    if selected is None:
        raise ValueError(
            "BLRDetector could not isolate one object inside that box. "
            "Try again with a tighter box around only one object."
        )

    contours, _ = cv2.findContours(
        selected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError(
            "No object contour was found inside the selected box."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    perimeter = cv2.arcLength(
        largest_contour,
        True,
    )

    epsilon = max(
        1.0,
        0.002
        * perimeter,
    )

    contour = cv2.approxPolyDP(
        largest_contour,
        epsilon,
        True,
    )

    localization_mask = (
        np.zeros_like(
            selected
        )
    )

    cv2.drawContours(
        localization_mask,
        [
            largest_contour
        ],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return (
        localization_mask,
        contour,
        selected_threshold,
        close_kernel_size,
    )


def create_manual_object_spec(
    image_contents,
    roi,
    manual_id,
):
    """
    Convert one user-drawn rectangle into one green manual object.

    The returned dictionary is JSON-serializable and can be stored
    directly in dcc.Store.
    """

    bgr = _decode_image(
        image_contents
    )

    image_h, image_w = (
        bgr.shape[:2]
    )

    (
        x0,
        y0,
        x1,
        y1,
    ) = _clip_manual_roi(
        roi,
        image_w,
        image_h,
    )

    crop = bgr[
        y0:y1,
        x0:x1,
    ]

    (
        local_mask,
        local_contour,
        local_threshold,
        close_kernel_size,
    ) = _manual_localization_mask(
        crop
    )

    points = (
        local_contour.reshape(
            -1,
            2,
        )
    )

    global_points = [
        [
            int(
                point[0]
                + x0
            ),
            int(
                point[1]
                + y0
            ),
        ]
        for point
        in points
    ]

    nonzero = cv2.findNonZero(
        local_mask
    )

    if nonzero is None:
        raise ValueError(
            "No usable manual object was found inside that box."
        )

    (
        bx,
        by,
        bw,
        bh,
    ) = cv2.boundingRect(
        nonzero
    )

    area = int(
        np.count_nonzero(
            local_mask
        )
    )

    return {
        "manual":
            True,
        "manual_id":
            int(
                manual_id
            ),
        "label":
            -int(
                manual_id
            ),
        "roi":
            {
                "x0":
                    int(
                        x0
                    ),
                "y0":
                    int(
                        y0
                    ),
                "x1":
                    int(
                        x1
                    ),
                "y1":
                    int(
                        y1
                    ),
            },
        "x":
            int(
                x0
                + bx
            ),
        "y":
            int(
                y0
                + by
            ),
        "width":
            int(
                bw
            ),
        "height":
            int(
                bh
            ),
        "area":
            area,
        "geometry_score":
            0.0,
        "manual_contours":
            [
                global_points
            ],
        "local_threshold":
            int(
                local_threshold
            ),
        "local_close_kernel":
            int(
                close_kernel_size
            ),
    }


def _candidate_replaced_by_manual(
    candidate,
    manual_object,
):
    """
    Decide whether an automatic fragment should disappear when the
    user repairs that physical object with Add Object.

    The manually drawn ROI is treated as the user's intent.
    """

    roi = manual_object[
        "roi"
    ]

    rx0 = float(
        roi["x0"]
    )

    ry0 = float(
        roi["y0"]
    )

    rx1 = float(
        roi["x1"]
    )

    ry1 = float(
        roi["y1"]
    )

    cx0 = float(
        candidate["x"]
    )

    cy0 = float(
        candidate["y"]
    )

    cx1 = (
        cx0
        + float(
            candidate["width"]
        )
    )

    cy1 = (
        cy0
        + float(
            candidate["height"]
        )
    )

    center_x = (
        cx0
        + cx1
    ) / 2.0

    center_y = (
        cy0
        + cy1
    ) / 2.0

    center_inside = (
        rx0
        <= center_x
        <= rx1
        and ry0
        <= center_y
        <= ry1
    )

    ix0 = max(
        rx0,
        cx0,
    )

    iy0 = max(
        ry0,
        cy0,
    )

    ix1 = min(
        rx1,
        cx1,
    )

    iy1 = min(
        ry1,
        cy1,
    )

    intersection = max(
        0.0,
        ix1 - ix0,
    ) * max(
        0.0,
        iy1 - iy0,
    )

    candidate_box_area = max(
        1.0,
        (
            cx1
            - cx0
        )
        * (
            cy1
            - cy0
        ),
    )

    overlap_fraction = (
        intersection
        / candidate_box_area
    )

    return (
        center_inside
        or overlap_fraction
        >= 0.25
    )


def _apply_manual_objects(
    analysis,
    manual_objects=None,
):
    """
    Replace automatic fragments covered by Add Object ROIs and append
    the repaired object as one green object.
    """

    manual_objects = (
        manual_objects
        or []
    )

    if not manual_objects:
        return analysis

    accepted = [
        dict(
            candidate
        )
        for candidate
        in analysis[
            "accepted"
        ]
    ]

    suspicious = [
        dict(
            candidate
        )
        for candidate
        in analysis[
            "suspicious"
        ]
    ]

    for manual_object in manual_objects:
        manual_item = dict(
            manual_object
        )

        accepted = [
            candidate
            for candidate
            in accepted
            if (
                candidate.get(
                    "manual"
                )
                or not _candidate_replaced_by_manual(
                    candidate,
                    manual_item,
                )
            )
        ]

        suspicious = [
            candidate
            for candidate
            in suspicious
            if not _candidate_replaced_by_manual(
                candidate,
                manual_item,
            )
        ]

        accepted.append(
            manual_item
        )

    accepted.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

    suspicious.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

    for index, candidate in enumerate(
        accepted,
        start=1,
    ):
        candidate[
            "object_number"
        ] = index

    for index, candidate in enumerate(
        suspicious,
        start=1,
    ):
        candidate[
            "suspicious_number"
        ] = index

    diagnostics = dict(
        analysis[
            "diagnostics"
        ]
    )

    diagnostics[
        "manual_object_count"
    ] = len(
        manual_objects
    )

    diagnostics[
        "accepted_count"
    ] = len(
        accepted
    )

    diagnostics[
        "suspicious_count"
    ] = len(
        suspicious
    )

    diagnostics[
        "difference"
    ] = (
        len(
            accepted
        )
        - diagnostics[
            "scanner_count"
        ]
    )

    diagnostics[
        "ready"
    ] = (
        len(
            accepted
        )
        == diagnostics[
            "scanner_count"
        ]
    )

    return {
        "bgr":
            analysis[
                "bgr"
            ],
        "labels":
            analysis[
                "labels"
            ],
        "accepted":
            accepted,
        "suspicious":
            suspicious,
        "diagnostics":
            diagnostics,
    }


def _manual_detector_labels(
    image_shape,
    manual_candidate,
):
    """
    Build a detector-label image so the existing validated
    ImageJ-compatible export function can be reused unchanged.
    """

    image_h, image_w = (
        image_shape[:2]
    )

    labels = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.int32,
    )

    contours = []

    for contour_points in manual_candidate.get(
        "manual_contours",
        [],
    ):
        if len(
            contour_points
        ) < 3:
            continue

        contour = np.asarray(
            contour_points,
            dtype=np.int32,
        ).reshape(
            -1,
            1,
            2,
        )

        contours.append(
            contour
        )

    if not contours:
        return labels

    cv2.drawContours(
        labels,
        contours,
        -1,
        1,
        thickness=cv2.FILLED,
    )

    return labels


def _build_manual_repair_export_mask(
    bgr,
    detector_labels,
    candidate,
    margin=4,
):
    """
    Export geometry for a user-created Add Object repair.

    A manual Add Object is an explicit statement from the reviewer that the
    selected region is ONE physical object. Re-running the global
    brightness/dark-limit segmentation can split that repaired object back
    into two pieces. For manual repairs, the filled manual contour therefore
    becomes the authoritative geometry.

    Original RGB pixels are still copied; only pure-zero pixels are omitted.
    This keeps dark metal pixels that were responsible for the original split
    while avoiding a large artificial black fill where possible.
    """

    image_h, image_w = bgr.shape[:2]
    x = int(candidate["x"])
    y = int(candidate["y"])
    w = int(candidate["width"])
    h = int(candidate["height"])
    label = int(candidate.get("label", 1))

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(image_w, x + w + margin)
    y1 = min(image_h, y + h + margin)

    crop = bgr[y0:y1, x0:x1]
    manual_mask = (detector_labels[y0:y1, x0:x1] == label)

    if not np.any(manual_mask):
        return None

    brightness = np.max(crop, axis=2)
    keep_mask = manual_mask & (brightness > 0)

    # If the source contains true zero-valued bridge pixels, keep the manual
    # geometry rather than allowing the repair to split again.
    count, _, _, _ = cv2.connectedComponentsWithStats(
        keep_mask.astype(np.uint8), connectivity=8
    )
    if count > 2:
        manual_count, _, _, _ = cv2.connectedComponentsWithStats(
            manual_mask.astype(np.uint8), connectivity=8
        )
        if manual_count == 2:
            keep_mask = manual_mask.copy()

    points = cv2.findNonZero(keep_mask.astype(np.uint8) * 255)
    if points is None:
        return None

    rx, ry, rw, rh = cv2.boundingRect(points)

    return {
        "crop_bgr": crop,
        "keep_mask": keep_mask,
        "outer_roi_mask": manual_mask,
        "roi_x": int(rx),
        "roi_y": int(ry),
        "roi_width": int(rw),
        "roi_height": int(rh),
        "crop_global_x": int(x0),
        "crop_global_y": int(y0),
        "global_x": int(x0 + rx),
        "global_y": int(y0 + ry),
        "strip_diagnostics": {
            "vertical_strips": [],
            "horizontal_strips": [],
            "manual_repair_authoritative": True,
        },
    }


# ---------------------------------------------------------
# Preview figure
# ---------------------------------------------------------

def _candidate_contours(
    labels,
    candidates,
    bgr=None,
    padding=3,
):
    """
    Build preview contours using only each candidate's local bbox.

    V0.53 created a full-image boolean mask once per object. For a shelf
    with ~100 objects this meant scanning the full RGB shelf ~100 times.
    V0.54 crops labels/RGB to the object's bbox first, so strip cleanup and
    contour extraction operate on a few thousand pixels instead of the
    entire shelf.
    """
    contour_x = []
    contour_y = []

    image_h, image_w = labels.shape[:2]

    for candidate in candidates:
        if candidate.get("manual"):
            contours = []
            contour_offsets = []

            for contour_points in candidate.get("manual_contours", []):
                if len(contour_points) < 2:
                    continue

                contours.append(
                    np.asarray(contour_points, dtype=np.int32).reshape(-1, 1, 2)
                )
                contour_offsets.append((0, 0))

        else:
            x = int(candidate["x"])
            y = int(candidate["y"])
            w = int(candidate["width"])
            h = int(candidate["height"])

            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(image_w, x + w + padding)
            y1 = min(image_h, y + h + padding)

            local_mask = (
                labels[y0:y1, x0:x1] == int(candidate["label"])
            )

            if bgr is not None and np.any(local_mask):
                local_bgr = bgr[y0:y1, x0:x1]
                local_mask, _ = _remove_long_dark_strip_artifacts(
                    crop_bgr=local_bgr,
                    keep_mask=local_mask,
                )

            component_mask = local_mask.astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour_offsets = [(x0, y0)] * len(contours)

        for contour, (offset_x, offset_y) in zip(contours, contour_offsets):
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(1.0, 0.002 * perimeter)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            points = simplified.reshape(-1, 2)
            if len(points) < 2:
                continue

            points = points.astype(np.int64, copy=False)
            points[:, 0] += int(offset_x)
            points[:, 1] += int(offset_y)
            points = np.vstack([points, points[0]])

            contour_x.extend(points[:, 0].tolist())
            contour_y.extend(points[:, 1].tolist())
            contour_x.append(None)
            contour_y.append(None)

    return contour_x, contour_y


def build_geometry_qc_preview(
    image_contents,
    metadata_df,
    manual_accepted_labels=None,
    manual_excluded_labels=None,
    manual_objects=None,
    qc_cache=None,
):
    """
    Green = included in download.
    Yellow = excluded by default.

    Optional review can move objects in either direction:
      yellow -> green
      green  -> yellow

    There are still only two visual states.
    """

    base_analysis = _analysis_from_geometry_qc_cache(
        image_contents,
        metadata_df,
        qc_cache,
    )

    analysis = _apply_manual_review(
        base_analysis,
        manual_accepted_labels,
        manual_excluded_labels,
    )

    analysis = _apply_manual_objects(
        analysis,
        manual_objects,
    )

    bgr = analysis[
        "bgr"
    ]

    labels = analysis[
        "labels"
    ]

    accepted = analysis[
        "accepted"
    ]

    suspicious = analysis[
        "suspicious"
    ]

    diagnostics = analysis[
        "diagnostics"
    ]

    (
        image_height,
        image_width,
    ) = bgr.shape[:2]

    contour_started = time.perf_counter()

    (
        accepted_x,
        accepted_y,
    ) = _candidate_contours(
        labels,
        accepted,
        bgr=bgr,
    )

    (
        suspicious_x,
        suspicious_y,
    ) = _candidate_contours(
        labels,
        suspicious,
        bgr=bgr,
    )

    diagnostics["preview_contour_seconds"] = float(
        time.perf_counter() - contour_started
    )

    figure = go.Figure()

    figure.add_layout_image(
        dict(
            source=image_contents,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=image_width,
            sizey=image_height,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            opacity=1,
            layer="below",
        )
    )

    figure.add_trace(
        go.Scattergl(
            x=accepted_x,
            y=accepted_y,
            mode="lines",
            line=dict(
                color="#22c55e",
                width=2,
            ),
            hoverinfo="skip",
            name="Accepted",
        )
    )

    # Clickable green objects.
    # customdata = ["green", component_label, visible_number]
    figure.add_trace(
        go.Scatter(
            x=[
                candidate["x"]
                + candidate["width"]
                / 2.0
                for candidate
                in accepted
            ],
            y=[
                candidate["y"]
                + candidate["height"]
                / 2.0
                for candidate
                in accepted
            ],
            mode="markers+text",
            text=[
                str(
                    candidate[
                        "object_number"
                    ]
                )
                for candidate
                in accepted
            ],
            textposition="top center",
            textfont=dict(
                color="white",
                size=9,
            ),
            marker=dict(
                size=30,
                color="#22c55e",
                opacity=0.10,
                line=dict(
                    color="#22c55e",
                    width=1,
                ),
            ),
            customdata=[
                [
                    (
                        "manual"
                        if candidate.get(
                            "manual"
                        )
                        else "green"
                    ),
                    int(
                        (
                            candidate[
                                "manual_id"
                            ]
                            if candidate.get(
                                "manual"
                            )
                            else candidate[
                                "label"
                            ]
                        )
                    ),
                    int(
                        candidate[
                            "object_number"
                        ]
                    ),
                ]
                for candidate
                in accepted
            ],
            hovertext=[
                (
                    (
                        f"Added object "
                        f"{candidate['object_number']}<br>"
                        f"{candidate['width']}×"
                        f"{candidate['height']} px<br>"
                        f"Area {candidate['area']}<br>"
                        f"Click to remove"
                    )
                    if candidate.get(
                        "manual"
                    )
                    else (
                        f"Accepted "
                        f"{candidate['object_number']}<br>"
                        f"{candidate['width']}×"
                        f"{candidate['height']} px<br>"
                        f"Area {candidate['area']}<br>"
                        f"Geometry score "
                        f"{candidate['geometry_score']:.2f}<br>"
                        f"Click to review"
                    )
                )
                for candidate
                in accepted
            ],
            hoverinfo="text",
            name="Accepted IDs",
        )
    )

    if suspicious:
        figure.add_trace(
            go.Scattergl(
                x=suspicious_x,
                y=suspicious_y,
                mode="lines",
                line=dict(
                    color="#f59e0b",
                    width=2,
                ),
                hoverinfo="skip",
                name="Suspicious",
            )
        )

        # Clickable yellow markers.
        # customdata = ["yellow", component_label, visible_S_number]
        figure.add_trace(
            go.Scatter(
                x=[
                    candidate["x"]
                    + candidate["width"]
                    / 2.0
                    for candidate
                    in suspicious
                ],
                y=[
                    candidate["y"]
                    + candidate["height"]
                    / 2.0
                    for candidate
                    in suspicious
                ],
                mode="markers+text",
                text=[
                    f"S{candidate['suspicious_number']}"
                    for candidate
                    in suspicious
                ],
                textposition="top center",
                textfont=dict(
                    color="#f59e0b",
                    size=10,
                ),
                marker=dict(
                    size=38,
                    color="#f59e0b",
                    opacity=0.18,
                    line=dict(
                        color="#f59e0b",
                        width=1,
                    ),
                ),
                customdata=[
                    [
                        "yellow",
                        int(
                            candidate[
                                "label"
                            ]
                        ),
                        int(
                            candidate[
                                "suspicious_number"
                            ]
                        ),
                    ]
                    for candidate
                    in suspicious
                ],
                hovertext=[
                    (
                        f"Suspicious S"
                        f"{candidate['suspicious_number']}<br>"
                        f"{candidate['width']}×"
                        f"{candidate['height']} px<br>"
                        f"Area {candidate['area']}<br>"
                        f"Geometry score "
                        f"{candidate['geometry_score']:.2f}<br>"
                        f"Click to select"
                    )
                    for candidate
                    in suspicious
                ],
                hoverinfo="text",
                name="Suspicious IDs",
            )
        )

    figure.update_xaxes(
        range=[
            0,
            image_width,
        ],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        constrain="range",
    )

    figure.update_yaxes(
        range=[
            image_height,
            0,
        ],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="range",
    )

    figure.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        dragmode="zoom",
        showlegend=False,
        clickmode="event",
        hovermode="closest",
        hoverdistance=60,
        plot_bgcolor="black",
        paper_bgcolor="black",
        autosize=True,
    )

    return (
        figure,
        accepted,
        suspicious,
        diagnostics,
    )


# ---------------------------------------------------------
# Final green-set export
# ---------------------------------------------------------

def generate_green_dataset_zip(
    image_contents,
    metadata_df,
    manual_accepted_labels,
    manual_excluded_labels,
    manual_objects,
    condition,
    material,
    target_size=320,
    export_brightness=10,
    dark_limit=9,
    qc_cache=None,
):
    """
    Export exactly the CURRENT GREEN SET after optional review.

    Yellow -> Green adds an object to the ZIP.
    Green  -> Yellow removes an object from the ZIP.

    Any object that is yellow at export time is excluded.

    Final image extraction uses the validated ImageJ-compatible path.
    """

    base_analysis = _analysis_from_geometry_qc_cache(
        image_contents,
        metadata_df,
        qc_cache,
    )

    analysis = _apply_manual_review(
        base_analysis,
        manual_accepted_labels,
        manual_excluded_labels,
    )

    analysis = _apply_manual_objects(
        analysis,
        manual_objects,
    )

    bgr = analysis[
        "bgr"
    ]

    detector_labels = analysis[
        "labels"
    ]

    accepted = analysis[
        "accepted"
    ]

    suspicious = analysis[
        "suspicious"
    ]

    qc_diagnostics = analysis[
        "diagnostics"
    ]

    condition_code = {
        "treated":
            "T",
        "non_treated":
            "NT",
    }.get(
        condition,
        condition,
    )

    material_code = {
        "copper":
            "Cu",
        "brass":
            "Brass",
        "stainless":
            "StSt",
        "grey":
            "Grey",
        "coins":
            "Coins",
    }.get(
        material,
        material,
    )

    metadata_rows = []

    skipped = 0

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        object_number = 1

        for candidate in accepted:
            if candidate.get(
                "manual"
            ):
                current_detector_labels = (
                    _manual_detector_labels(
                        bgr.shape,
                        candidate,
                    )
                )

                current_candidate = dict(
                    candidate
                )

                current_candidate[
                    "label"
                ] = 1

            else:
                current_detector_labels = (
                    detector_labels
                )

                current_candidate = (
                    candidate
                )

            if candidate.get("manual"):
                export_data = _build_manual_repair_export_mask(
                    bgr=bgr,
                    detector_labels=current_detector_labels,
                    candidate=current_candidate,
                )
            else:
                export_data = (
                    _build_imagej_export_mask(
                        bgr=bgr,
                        detector_labels=
                            current_detector_labels,
                        candidate=
                            current_candidate,
                        brightness_min=
                            export_brightness,
                        dark_limit=
                            dark_limit,
                    )
                )

            if export_data is None:
                skipped += 1
                continue

            # Final single-piece rule:
            # - tiny disconnected fragments are discarded
            # - a similarly sized second component is exported separately
            # - every PNG therefore contains one retained component
            piece_exports = (
                _split_export_into_single_piece_regions(
                    export_data,
                    small_fragment_ratio=0.10,
                    split_fragment_ratio=0.35,
                    min_fragment_area_px=12,
                )
            )

            if not piece_exports:
                skipped += 1
                continue

            for piece_export in piece_exports:
                roi_x = piece_export[
                    "roi_x"
                ]

                roi_y = piece_export[
                    "roi_y"
                ]

                width = piece_export[
                    "roi_width"
                ]

                height = piece_export[
                    "roi_height"
                ]

                if (
                    width > target_size
                    or height > target_size
                ):
                    skipped += 1
                    continue

                crop_bgr = piece_export[
                    "crop_bgr"
                ]

                keep_mask = piece_export[
                    "keep_mask"
                ]

                roi_bgr = crop_bgr[
                    roi_y:roi_y + height,
                    roi_x:roi_x + width,
                ]

                roi_keep = keep_mask[
                    roi_y:roi_y + height,
                    roi_x:roi_x + width,
                ]

                canvas = np.zeros(
                    (
                        target_size,
                        target_size,
                        3,
                    ),
                    dtype=np.uint8,
                )

                # OpenCV BGR -> pure RGB blue background.
                canvas[:, :] = [
                    255,
                    0,
                    0,
                ]

                paste_x = (
                    target_size
                    - width
                ) // 2

                paste_y = (
                    target_size
                    - height
                ) // 2

                target_region = canvas[
                    paste_y:paste_y + height,
                    paste_x:paste_x + width,
                ]

                target_region[
                    roi_keep
                ] = roi_bgr[
                    roi_keep
                ]

                output_filename = (
                    f"{material_code}_"
                    f"{condition_code}_"
                    f"{object_number:03d}.png"
                )

                success, png_buffer = (
                    cv2.imencode(
                        ".png",
                        canvas,
                    )
                )

                if not success:
                    skipped += 1
                    continue

                zip_file.writestr(
                    f"images/{output_filename}",
                    png_buffer.tobytes(),
                )

                metadata_rows.append(
                    {
                        "object_id":
                            object_number,
                        "condition":
                            condition,
                        "material":
                            material,
                        "filename":
                            output_filename,
                        "x":
                            piece_export[
                                "global_x"
                            ],
                        "y":
                            piece_export[
                                "global_y"
                            ],
                        "width_px":
                            width,
                        "height_px":
                            height,
                        "geometry_score":
                            candidate[
                                "geometry_score"
                            ],
                        "detection_brightness":
                            qc_diagnostics[
                                "brightness_min"
                            ],
                        "detection_kernel":
                            qc_diagnostics[
                                "kernel_size"
                            ],
                        "export_brightness":
                            export_brightness,
                        "dark_limit":
                            dark_limit,
                        "status":
                            "exported",
                    }
                )

                object_number += 1

        csv_buffer = io.StringIO()

        fieldnames = [
            "object_id",
            "condition",
            "material",
            "filename",
            "x",
            "y",
            "width_px",
            "height_px",
            "geometry_score",
            "detection_brightness",
            "detection_kernel",
            "export_brightness",
            "dark_limit",
            "status",
        ]

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

        zip_file.writestr(
            "metadata.csv",
            csv_buffer.getvalue(),
        )

    zip_buffer.seek(
        0
    )

    diagnostics = {
        "scanner_count":
            len(
                metadata_df
            ),
        "green_count":
            len(
                accepted
            ),
        "yellow_remaining":
            len(
                suspicious
            ),
        "exported":
            len(
                metadata_rows
            ),
        "skipped":
            skipped,
        "brightness_min":
            qc_diagnostics[
                "brightness_min"
            ],
        "kernel_size":
            qc_diagnostics[
                "kernel_size"
            ],
    }

    return (
        zip_buffer.getvalue(),
        diagnostics,
    )
