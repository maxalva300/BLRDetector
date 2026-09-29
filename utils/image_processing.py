import base64
import io
import csv
import zipfile

import cv2
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------
# Internal segmentation function
# ---------------------------------------------------------

def _segment_objects(
    bgr,
    brightness_min,
    kernel_size,
    min_area,
    max_area,
):
    """
    Global BLRDetector segmentation used for object localization.

    This stage is optimized for finding/counting objects.
    It is NOT used directly as the final export mask.
    """

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV,
    )

    brightness = hsv[:, :, 2]

    # Detect artificial grey shelf regions
    channel_max = np.max(bgr, axis=2)
    channel_min = np.min(bgr, axis=2)
    channel_mean = np.mean(bgr, axis=2)

    neutral_difference = (
        channel_max - channel_min
    )

    shelf_mask = (
        (neutral_difference <= 3)
        & (channel_mean >= 115)
        & (channel_mean <= 140)
    )

    foreground = (
        (brightness >= brightness_min)
        & (~shelf_mask)
    )

    binary = (
        foreground.astype(np.uint8) * 255
    )

    # Morphology is used ONLY for robust localization
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
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

    for label in range(1, number_labels):
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
                "label": label,
                "x": int(
                    stats[
                        label,
                        cv2.CC_STAT_LEFT,
                    ]
                ),
                "y": int(
                    stats[
                        label,
                        cv2.CC_STAT_TOP,
                    ]
                ),
                "width": int(
                    stats[
                        label,
                        cv2.CC_STAT_WIDTH,
                    ]
                ),
                "height": int(
                    stats[
                        label,
                        cv2.CC_STAT_HEIGHT,
                    ]
                ),
                "area": area,
            }
        )

    return labels, candidates


def _find_best_detection(
    bgr,
    target_count,
    min_area,
    max_area,
):
    """
    Find localization parameters whose candidate count is closest
    to the object count reported by the scanner metadata.
    """

    brightness_values = [
        8, 10, 12, 15, 18,
        22, 26, 30, 35, 40,
    ]

    kernel_values = [
        3, 5, 7, 9, 11,
    ]

    # Baseline when metadata is unavailable
    if target_count is None:
        brightness_min = 10
        kernel_size = 5

        labels, candidates = _segment_objects(
            bgr,
            brightness_min,
            kernel_size,
            min_area,
            max_area,
        )

        return (
            brightness_min,
            kernel_size,
            labels,
            candidates,
        )

    best_score = None
    best_result = None

    for brightness_min in brightness_values:
        for kernel_size in kernel_values:
            (
                trial_labels,
                trial_candidates,
            ) = _segment_objects(
                bgr,
                brightness_min,
                kernel_size,
                min_area,
                max_area,
            )

            detected_count = len(
                trial_candidates
            )

            count_difference = abs(
                detected_count - target_count
            )

            # Main goal: match scanner object count.
            # Secondary penalty keeps parameters near baseline.
            score = (
                count_difference * 1000
                + abs(brightness_min - 10) * 2
                + abs(kernel_size - 5)
            )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score

                best_result = (
                    brightness_min,
                    kernel_size,
                    trial_labels,
                    trial_candidates,
                )

    return best_result


# ---------------------------------------------------------
# Main preview function
# ---------------------------------------------------------

def build_detection_preview(
    image_contents,
    target_count=None,
    min_area=20,
    max_area=8000,
):
    """
    Build the interactive detection preview.

    The original PNG is displayed as a browser-native background.
    Only contours and object IDs are rendered as Plotly vectors.
    """

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

    image_height, image_width = (
        bgr.shape[:2]
    )

    (
        selected_brightness,
        selected_kernel,
        labels,
        candidates,
    ) = _find_best_detection(
        bgr,
        target_count,
        min_area,
        max_area,
    )

    candidates.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

    contour_x = []
    contour_y = []

    number_x = []
    number_y = []
    number_text = []

    for object_number, obj in enumerate(
        candidates,
        start=1,
    ):
        obj["object_number"] = (
            object_number
        )

        component_mask = (
            labels == obj["label"]
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            perimeter = cv2.arcLength(
                contour,
                True,
            )

            epsilon = max(
                1.0,
                0.002 * perimeter,
            )

            simplified = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            points = simplified.reshape(
                -1,
                2,
            )

            if len(points) < 2:
                continue

            points = np.vstack(
                [
                    points,
                    points[0],
                ]
            )

            contour_x.extend(
                points[:, 0].tolist()
            )

            contour_y.extend(
                points[:, 1].tolist()
            )

            contour_x.append(None)
            contour_y.append(None)

        number_x.append(
            obj["x"]
        )

        number_y.append(
            max(
                0,
                obj["y"] - 5,
            )
        )

        number_text.append(
            str(object_number)
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
            x=contour_x,
            y=contour_y,
            mode="lines",
            line=dict(
                color="cyan",
                width=2,
            ),
            hoverinfo="skip",
            name="Candidates",
        )
    )

    figure.add_trace(
        go.Scattergl(
            x=number_x,
            y=number_y,
            mode="text",
            text=number_text,
            textfont=dict(
                color="white",
                size=11,
            ),
            hoverinfo="skip",
            name="Object ID",
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
        constrain="domain",
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
    )

    figure.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        height=700,
        dragmode="pan",
        showlegend=False,
        plot_bgcolor="black",
        uirevision="blr-preview",
    )

    diagnostics = {
        "brightness_min":
            selected_brightness,
        "kernel_size":
            selected_kernel,
        "detected_count":
            len(candidates),
        "target_count":
            target_count,
    }

    return (
        figure,
        candidates,
        diagnostics,
    )


# ---------------------------------------------------------
# Repeating dark strip artifact cleanup
# ---------------------------------------------------------

def _contiguous_true_runs(flags):
    """Return inclusive (start, end) runs from a 1-D boolean array."""

    indices = np.flatnonzero(flags)

    if len(indices) == 0:
        return []

    runs = []
    start = int(indices[0])
    previous = int(indices[0])

    for value in indices[1:]:
        value = int(value)

        if value == previous + 1:
            previous = value
            continue

        runs.append((start, previous))
        start = value
        previous = value

    runs.append((start, previous))

    return runs


def _remove_long_dark_strip_artifacts(
    crop_bgr,
    keep_mask,
    occupancy_min=0.88,
    core_median_brightness_max=20,
    core_median_chroma_max=9,
    min_strip_thickness=3,
    max_strip_thickness=20,
    max_strip_fraction=0.45,
    fringe_occupancy_min=0.25,
    fringe_median_brightness_max=25,
    fringe_median_chroma_max=12,
    max_fringe_expand=8,
    remove_brightness_max=28,
    remove_chroma_max=14,
    halo_px=4,
    halo_brightness_max=16,
    halo_chroma_max=12,
    residual_halo_expand=7,
    residual_brightness_max=17,
    residual_chroma_max=10,
):
    """
    Remove the recurring long, dark scanner/shelf strip artifact without
    raising the global ImageJ brightness threshold.

    The artifact pattern observed in BLR batches is:
      - long and almost perfectly vertical or horizontal
      - narrow relative to the object bbox
      - dark / nearly neutral grey
      - occupies most of the object bbox height or width

    Only pixels that also look like the dark strip are removed. Brighter
    or more chromatic pixels are preserved so a real metal object crossing
    the strip is disturbed as little as possible.

    Returns
    -------
    cleaned_mask : bool ndarray
        Updated final copy mask.
    diagnostics : dict
        Number and location of detected strips, useful for later QC.
    """

    cleaned = keep_mask.astype(bool).copy()

    points = cv2.findNonZero(
        cleaned.astype(np.uint8) * 255
    )

    if points is None:
        return cleaned, {
            "vertical_strips": [],
            "horizontal_strips": [],
        }

    bx, by, bw, bh = cv2.boundingRect(points)

    # Very small outputs cannot contain the repeated shelf-strip pattern.
    if bw < 8 or bh < 8:
        return cleaned, {
            "vertical_strips": [],
            "horizontal_strips": [],
        }

    channel_max = np.max(crop_bgr, axis=2).astype(np.int16)
    channel_min = np.min(crop_bgr, axis=2).astype(np.int16)
    chroma = channel_max - channel_min
    brightness = channel_max

    local_mask = cleaned[
        by:by + bh,
        bx:bx + bw,
    ]

    local_brightness = brightness[
        by:by + bh,
        bx:bx + bw,
    ]

    local_chroma = chroma[
        by:by + bh,
        bx:bx + bw,
    ]

    diagnostics = {
        "vertical_strips": [],
        "horizontal_strips": [],
    }

    # -------------------------
    # Vertical strip detection
    # -------------------------
    column_occupancy = np.zeros(bw, dtype=np.float32)
    column_median_brightness = np.full(
        bw,
        255.0,
        dtype=np.float32,
    )
    column_median_chroma = np.full(
        bw,
        255.0,
        dtype=np.float32,
    )

    for column in range(bw):
        column_mask = local_mask[:, column]

        column_occupancy[column] = (
            np.count_nonzero(column_mask)
            / float(bh)
        )

        if np.any(column_mask):
            column_median_brightness[column] = np.median(
                local_brightness[:, column][column_mask]
            )
            column_median_chroma[column] = np.median(
                local_chroma[:, column][column_mask]
            )

    vertical_core = (
        (column_occupancy >= occupancy_min)
        & (
            column_median_brightness
            <= core_median_brightness_max
        )
        & (
            column_median_chroma
            <= core_median_chroma_max
        )
    )

    for start, end in _contiguous_true_runs(
        vertical_core
    ):
        thickness = end - start + 1

        if thickness < min_strip_thickness:
            continue

        if thickness > max_strip_thickness:
            continue

        if thickness > max_strip_fraction * bw:
            # Protect genuine long/thin objects whose whole bbox is narrow.
            continue

        expanded_start = start
        expanded_end = end

        for _ in range(max_fringe_expand):
            next_column = expanded_start - 1

            if next_column < 0:
                break

            if not (
                column_occupancy[next_column]
                >= fringe_occupancy_min
                and column_median_brightness[next_column]
                <= fringe_median_brightness_max
                and column_median_chroma[next_column]
                <= fringe_median_chroma_max
            ):
                break

            expanded_start = next_column

        for _ in range(max_fringe_expand):
            next_column = expanded_end + 1

            if next_column >= bw:
                break

            if not (
                column_occupancy[next_column]
                >= fringe_occupancy_min
                and column_median_brightness[next_column]
                <= fringe_median_brightness_max
                and column_median_chroma[next_column]
                <= fringe_median_chroma_max
            ):
                break

            expanded_end = next_column

        expanded_thickness = (
            expanded_end - expanded_start + 1
        )

        if expanded_thickness > 0.65 * bw:
            continue

        # There must be meaningful object content outside the strip. This
        # avoids deleting an actual thin, dark metal object.
        outside = local_mask.copy()
        outside[:, expanded_start:expanded_end + 1] = False

        outside_pixels = np.count_nonzero(outside)
        total_pixels = np.count_nonzero(local_mask)

        if outside_pixels < max(
            10,
            int(0.05 * total_pixels),
        ):
            continue

        strip_region = np.zeros_like(
            cleaned,
            dtype=bool,
        )

        global_x0 = bx + expanded_start
        global_x1 = bx + expanded_end

        strip_region[
            by:by + bh,
            global_x0:global_x1 + 1,
        ] = True

        # Remove only pixels that match the observed dark/neutral strip
        # appearance. Brighter metal pixels crossing the strip are kept.
        artifact_pixels = (
            strip_region
            & cleaned
            & (brightness <= remove_brightness_max)
            & (chroma <= remove_chroma_max)
        )

        cleaned[artifact_pixels] = False

        # The scanner strip has anti-aliased / serrated fringe pixels.
        # They can have very low occupancy per column, so the projection
        # detector may identify the strip core but leave dotted remnants.
        # Remove ONLY ultra-dark, near-neutral pixels in a small halo
        # around the detected strip. Real metal pixels that are brighter
        # or more chromatic are preserved.
        halo_region = np.zeros_like(
            cleaned,
            dtype=bool,
        )

        halo_x0 = max(
            0,
            global_x0 - halo_px,
        )
        halo_x1 = min(
            cleaned.shape[1] - 1,
            global_x1 + halo_px,
        )

        halo_region[
            by:by + bh,
            halo_x0:halo_x1 + 1,
        ] = True

        halo_pixels = (
            halo_region
            & cleaned
            & (brightness <= halo_brightness_max)
            & (chroma <= halo_chroma_max)
        )

        cleaned[halo_pixels] = False

        diagnostics["vertical_strips"].append(
            {
                "x0": int(global_x0),
                "x1": int(global_x1),
                "halo_x0": int(halo_x0),
                "halo_x1": int(halo_x1),
                "removed_pixels": int(
                    np.count_nonzero(artifact_pixels)
                    + np.count_nonzero(halo_pixels)
                ),
            }
        )

    # ---------------------------
    # Horizontal strip detection
    # ---------------------------
    # Recompute the local mask after vertical cleanup.
    local_mask = cleaned[
        by:by + bh,
        bx:bx + bw,
    ]

    row_occupancy = np.zeros(bh, dtype=np.float32)
    row_median_brightness = np.full(
        bh,
        255.0,
        dtype=np.float32,
    )
    row_median_chroma = np.full(
        bh,
        255.0,
        dtype=np.float32,
    )

    for row in range(bh):
        row_mask = local_mask[row, :]

        row_occupancy[row] = (
            np.count_nonzero(row_mask)
            / float(bw)
        )

        if np.any(row_mask):
            row_median_brightness[row] = np.median(
                local_brightness[row, :][row_mask]
            )
            row_median_chroma[row] = np.median(
                local_chroma[row, :][row_mask]
            )

    horizontal_core = (
        (row_occupancy >= occupancy_min)
        & (
            row_median_brightness
            <= core_median_brightness_max
        )
        & (
            row_median_chroma
            <= core_median_chroma_max
        )
    )

    for start, end in _contiguous_true_runs(
        horizontal_core
    ):
        thickness = end - start + 1

        if thickness < min_strip_thickness:
            continue

        if thickness > max_strip_thickness:
            continue

        if thickness > max_strip_fraction * bh:
            continue

        expanded_start = start
        expanded_end = end

        for _ in range(max_fringe_expand):
            next_row = expanded_start - 1

            if next_row < 0:
                break

            if not (
                row_occupancy[next_row]
                >= fringe_occupancy_min
                and row_median_brightness[next_row]
                <= fringe_median_brightness_max
                and row_median_chroma[next_row]
                <= fringe_median_chroma_max
            ):
                break

            expanded_start = next_row

        for _ in range(max_fringe_expand):
            next_row = expanded_end + 1

            if next_row >= bh:
                break

            if not (
                row_occupancy[next_row]
                >= fringe_occupancy_min
                and row_median_brightness[next_row]
                <= fringe_median_brightness_max
                and row_median_chroma[next_row]
                <= fringe_median_chroma_max
            ):
                break

            expanded_end = next_row

        expanded_thickness = (
            expanded_end - expanded_start + 1
        )

        if expanded_thickness > 0.65 * bh:
            continue

        outside = local_mask.copy()
        outside[expanded_start:expanded_end + 1, :] = False

        outside_pixels = np.count_nonzero(outside)
        total_pixels = np.count_nonzero(local_mask)

        if outside_pixels < max(
            10,
            int(0.05 * total_pixels),
        ):
            continue

        strip_region = np.zeros_like(
            cleaned,
            dtype=bool,
        )

        global_y0 = by + expanded_start
        global_y1 = by + expanded_end

        strip_region[
            global_y0:global_y1 + 1,
            bx:bx + bw,
        ] = True

        artifact_pixels = (
            strip_region
            & cleaned
            & (brightness <= remove_brightness_max)
            & (chroma <= remove_chroma_max)
        )

        cleaned[artifact_pixels] = False

        halo_region = np.zeros_like(
            cleaned,
            dtype=bool,
        )

        halo_y0 = max(
            0,
            global_y0 - halo_px,
        )
        halo_y1 = min(
            cleaned.shape[0] - 1,
            global_y1 + halo_px,
        )

        halo_region[
            halo_y0:halo_y1 + 1,
            bx:bx + bw,
        ] = True

        halo_pixels = (
            halo_region
            & cleaned
            & (brightness <= halo_brightness_max)
            & (chroma <= halo_chroma_max)
        )

        cleaned[halo_pixels] = False

        diagnostics["horizontal_strips"].append(
            {
                "y0": int(global_y0),
                "y1": int(global_y1),
                "halo_y0": int(halo_y0),
                "halo_y1": int(halo_y1),
                "removed_pixels": int(
                    np.count_nonzero(artifact_pixels)
                    + np.count_nonzero(halo_pixels)
                ),
            }
        )

    # -----------------------------------------------------
    # Residual strip-tail cleanup (V0.56)
    # -----------------------------------------------------
    # After the main strip has been removed, a short serrated/dotted tail
    # can remain attached to the object. It is no longer long enough to
    # satisfy the original strip projection test, but it still lies next
    # to a strip that was already detected with high confidence.
    #
    # Use that provenance instead of changing the global threshold:
    # remove ONLY ultra-dark, near-neutral pixels inside a slightly wider
    # corridor around an already detected strip. This is local and cheap.
    residual_corridor = np.zeros_like(cleaned, dtype=bool)

    for item in diagnostics.get("vertical_strips", []):
        x0 = max(0, int(item.get("halo_x0", item["x0"])) - residual_halo_expand)
        x1 = min(
            cleaned.shape[1] - 1,
            int(item.get("halo_x1", item["x1"])) + residual_halo_expand,
        )
        residual_corridor[by:by + bh, x0:x1 + 1] = True

    for item in diagnostics.get("horizontal_strips", []):
        y0 = max(0, int(item.get("halo_y0", item["y0"])) - residual_halo_expand)
        y1 = min(
            cleaned.shape[0] - 1,
            int(item.get("halo_y1", item["y1"])) + residual_halo_expand,
        )
        residual_corridor[y0:y1 + 1, bx:bx + bw] = True

    residual_pixels = (
        residual_corridor
        & cleaned
        & (brightness <= residual_brightness_max)
        & (chroma <= residual_chroma_max)
    )

    residual_removed = int(np.count_nonzero(residual_pixels))
    if residual_removed:
        cleaned[residual_pixels] = False

    diagnostics["residual_strip_pixels_removed"] = residual_removed

    return cleaned, diagnostics


# ---------------------------------------------------------
# ImageJ-compatible export mask
# ---------------------------------------------------------

def _build_imagej_export_mask(
    bgr,
    detector_labels,
    candidate,
    brightness_min=10,
    dark_limit=9,
    margin=4,
):
    """
    Build the high-fidelity export mask using the logic of
    KI_Objekt_Selektion + KI_Objektetransfer_320x320_RGB_.

    BLRDetector is used only to locate the object.
    Export uses:
      - original RGB/BGR pixels
      - HSB/HSV brightness threshold >= 10
      - NO morphology
      - an outer ROI
      - dark-hole removal: R,G,B <= dark_limit stay blue
    """

    image_h, image_w = bgr.shape[:2]

    x = candidate["x"]
    y = candidate["y"]
    w = candidate["width"]
    h = candidate["height"]
    label = candidate["label"]

    x0 = max(
        0,
        x - margin,
    )

    y0 = max(
        0,
        y - margin,
    )

    x1 = min(
        image_w,
        x + w + margin,
    )

    y1 = min(
        image_h,
        y + h + margin,
    )

    crop = bgr[
        y0:y1,
        x0:x1,
    ]

    # ImageJ equivalent:
    # RGB -> HSB -> Brightness >= 10
    # OpenCV HSV V channel is the brightness/value channel.
    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    brightness = hsv[:, :, 2]

    raw_mask = (
        brightness >= brightness_min
    ).astype(np.uint8) * 255

    # Automated replacement for manual grey-shelf cleanup
    channel_max = np.max(
        crop,
        axis=2,
    )

    channel_min = np.min(
        crop,
        axis=2,
    )

    channel_mean = np.mean(
        crop,
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

    raw_mask[
        shelf_mask
    ] = 0

    # Determine which low-threshold components belong
    # to this BLRDetector candidate.
    detector_crop = (
        detector_labels[
            y0:y1,
            x0:x1,
        ] == label
    ).astype(np.uint8)

    (
        component_count,
        component_labels,
        _,
        _,
    ) = cv2.connectedComponentsWithStats(
        raw_mask,
        connectivity=8,
    )

    selected_components = []

    for component_id in range(
        1,
        component_count,
    ):
        component_mask = (
            component_labels
            == component_id
        )

        overlap = np.count_nonzero(
            component_mask
            & (detector_crop > 0)
        )

        if overlap > 0:
            selected_components.append(
                component_id
            )

    if not selected_components:
        return None

    selected_mask = np.isin(
        component_labels,
        selected_components,
    ).astype(np.uint8) * 255

    # Outer ROI only, matching the ImageJ ROI behavior.
    # Internal holes are handled separately by dark_limit.
    contours, _ = cv2.findContours(
        selected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    outer_roi = np.zeros_like(
        selected_mask
    )

    cv2.drawContours(
        outer_roi,
        contours,
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # Exact ImageJ dark-hole rule:
    # if R <= darkLimit AND G <= darkLimit AND B <= darkLimit,
    # do not copy that pixel; it remains blue.
    b = crop[:, :, 0]
    g = crop[:, :, 1]
    r = crop[:, :, 2]

    dark_pixels = (
        (r <= dark_limit)
        & (g <= dark_limit)
        & (b <= dark_limit)
    )

    keep_mask = (
        (outer_roi > 0)
        & (~dark_pixels)
    )

    # Final QC cleanup for the recurring long dark scanner/shelf strip.
    # This is deliberately applied AFTER the validated ImageJ-compatible
    # extraction logic, so the main detector and threshold remain unchanged.
    keep_mask, strip_diagnostics = (
        _remove_long_dark_strip_artifacts(
            crop_bgr=crop,
            keep_mask=keep_mask,
        )
    )

    # Use the cleaned copy mask for the final bbox so a removed strip does
    # not keep the object artificially wide/tall or off-centre.
    roi_points = cv2.findNonZero(
        keep_mask.astype(np.uint8) * 255
    )

    if roi_points is None:
        return None

    rx, ry, rw, rh = cv2.boundingRect(
        roi_points
    )

    return {
        "crop_bgr":
            crop,
        "keep_mask":
            keep_mask,
        "outer_roi_mask":
            (outer_roi > 0),
        "roi_x":
            rx,
        "roi_y":
            ry,
        "roi_width":
            rw,
        "roi_height":
            rh,
        "crop_global_x":
            x0,
        "crop_global_y":
            y0,
        "global_x":
            x0 + rx,
        "global_y":
            y0 + ry,
        "strip_diagnostics":
            strip_diagnostics,
    }


# ---------------------------------------------------------
# Final single-piece cleanup for dataset export
# ---------------------------------------------------------

def _split_export_into_single_piece_regions(
    export_data,
    small_fragment_ratio=0.10,
    split_fragment_ratio=0.35,
    min_fragment_area_px=12,
    final_island_min_px=8,
):
    """
    V0.55 provenance-aware final cleanup.

    IMPORTANT DESIGN RULE
    ---------------------
    One reviewed detector candidate remains one output PNG.

    The final ImageJ-compatible operations (dark_limit and scanner-strip
    cleanup) are allowed to make one physical object look disconnected.
    Therefore we no longer split a candidate into multiple PNGs merely
    because the *final* mask contains several connected components.

    What we remove automatically:
      1) tiny detached islands (< final_island_min_px), and
      2) components that still look like residual pieces of a detected
         vertical/horizontal scanner strip.

    What we preserve:
      - significant disconnected regions that remain after dark_limit or
        strip cleanup. They stay in the SAME PNG because they may be parts
        of one physical object that our processing visually separated.

    The old ratio parameters are retained in the signature for backwards
    compatibility with geometry_qc.py, but V0.55 intentionally does not use
    the 10%-35%-split rule as an object-identity decision.
    """

    crop_bgr = export_data["crop_bgr"]
    keep_mask = export_data["keep_mask"].astype(bool)
    strip_diagnostics = export_data.get("strip_diagnostics", {}) or {}

    if not np.any(keep_mask):
        return []

    # Work only in the current local object bbox. This keeps the final QC
    # extremely cheap even for large scanner shelves.
    points = cv2.findNonZero(keep_mask.astype(np.uint8) * 255)
    if points is None:
        return []

    bx, by, bw, bh = cv2.boundingRect(points)

    local_keep = keep_mask[
        by:by + bh,
        bx:bx + bw,
    ].astype(np.uint8)

    local_bgr = crop_bgr[
        by:by + bh,
        bx:bx + bw,
    ]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        local_keep,
        connectivity=8,
    )

    if count <= 1:
        return []

    channel_max = np.max(local_bgr, axis=2).astype(np.int16)
    channel_min = np.min(local_bgr, axis=2).astype(np.int16)
    brightness = channel_max
    chroma = channel_max - channel_min

    cleaned_local = np.zeros_like(local_keep, dtype=bool)

    removed_tiny = 0
    removed_strip_residue = 0
    preserved_components = 0

    # Convert detected strip/halo coordinates from crop coordinates to this
    # small final bbox. These zones are used only to identify *residual*
    # strip fragments, not to erase arbitrary object pixels.
    vertical_zones = []
    for item in strip_diagnostics.get("vertical_strips", []):
        x0 = int(item.get("halo_x0", item.get("x0", -1))) - bx
        x1 = int(item.get("halo_x1", item.get("x1", -1))) - bx
        x0 = max(0, x0)
        x1 = min(bw - 1, x1)
        if x1 >= x0:
            vertical_zones.append((x0, x1))

    horizontal_zones = []
    for item in strip_diagnostics.get("horizontal_strips", []):
        y0 = int(item.get("halo_y0", item.get("y0", -1))) - by
        y1 = int(item.get("halo_y1", item.get("y1", -1))) - by
        y0 = max(0, y0)
        y1 = min(bh - 1, y1)
        if y1 >= y0:
            horizontal_zones.append((y0, y1))

    component_records = []

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area <= 0:
            continue

        component = (labels == component_id)

        cx = int(stats[component_id, cv2.CC_STAT_LEFT])
        cy = int(stats[component_id, cv2.CC_STAT_TOP])
        cw = int(stats[component_id, cv2.CC_STAT_WIDTH])
        ch = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        component_records.append({
            "id": component_id,
            "area": area,
            "mask": component,
            "x": cx,
            "y": cy,
            "w": cw,
            "h": ch,
        })

    if not component_records:
        return []

    component_records.sort(key=lambda item: item["area"], reverse=True)
    largest_area = max(component_records[0]["area"], 1)

    for index, record in enumerate(component_records):
        component = record["mask"]
        area = record["area"]

        # Always protect the dominant object component. It can legitimately
        # be very small in unusual batches.
        is_largest = (index == 0)

        if (not is_largest) and area < final_island_min_px:
            record["looks_like_strip_residue"] = False
            continue

        comp_brightness = brightness[component]
        comp_chroma = chroma[component]

        median_brightness = (
            float(np.median(comp_brightness))
            if comp_brightness.size else 255.0
        )
        median_chroma = (
            float(np.median(comp_chroma))
            if comp_chroma.size else 255.0
        )

        # Residual scanner-strip pieces are expected to be:
        #   - mostly inside the already detected strip halo,
        #   - dark / near-neutral, and
        #   - elongated in the direction of that strip.
        # A real metal fragment crossing the strip is usually brighter,
        # more chromatic, or not strongly elongated, so it is preserved.
        looks_like_strip_residue = False

        for zx0, zx1 in vertical_zones:
            zone = np.zeros_like(component, dtype=bool)
            zone[:, zx0:zx1 + 1] = True
            overlap = np.count_nonzero(component & zone)
            overlap_ratio = overlap / float(max(area, 1))
            zone_width = zx1 - zx0 + 1

            elongated_vertical = (
                record["h"] >= max(6, int(1.6 * record["w"]))
                and record["w"] <= max(4, int(2.0 * zone_width))
            )

            if (
                overlap_ratio >= 0.55
                and elongated_vertical
                and median_brightness <= 34
                and median_chroma <= 18
            ):
                looks_like_strip_residue = True
                break

        if not looks_like_strip_residue:
            for zy0, zy1 in horizontal_zones:
                zone = np.zeros_like(component, dtype=bool)
                zone[zy0:zy1 + 1, :] = True
                overlap = np.count_nonzero(component & zone)
                overlap_ratio = overlap / float(max(area, 1))
                zone_height = zy1 - zy0 + 1

                elongated_horizontal = (
                    record["w"] >= max(6, int(1.6 * record["h"]))
                    and record["h"] <= max(4, int(2.0 * zone_height))
                )

                if (
                    overlap_ratio >= 0.55
                    and elongated_horizontal
                    and median_brightness <= 34
                    and median_chroma <= 18
                ):
                    looks_like_strip_residue = True
                    break

        record["looks_like_strip_residue"] = bool(looks_like_strip_residue)

    # Remove strip-like residual components only when there is at least one
    # plausible non-strip object component to retain. This lets us remove a
    # long artifact even when the artifact happens to contain more pixels
    # than a small metal piece, while still protecting genuinely long/thin
    # objects in candidates where no alternative object component exists.
    has_non_strip_component = any(
        not record.get("looks_like_strip_residue", False)
        for record in component_records
        if record["area"] >= final_island_min_px
    )

    cleaned_local[:] = False
    preserved_components = 0

    for index, record in enumerate(component_records):
        component = record["mask"]
        area = record["area"]
        is_largest = (index == 0)

        if (not is_largest) and area < final_island_min_px:
            removed_tiny += 1
            continue

        if (
            record.get("looks_like_strip_residue", False)
            and has_non_strip_component
        ):
            removed_strip_residue += 1
            continue

        cleaned_local |= component
        preserved_components += 1

    # Safety: never erase the candidate completely.
    if not np.any(cleaned_local):
        largest_id = component_records[0]["id"]
        cleaned_local = (labels == largest_id)
        preserved_components = 1

    cleaned_keep = np.zeros_like(keep_mask, dtype=bool)
    cleaned_keep[
        by:by + bh,
        bx:bx + bw,
    ] = cleaned_local

    final_points = cv2.findNonZero(
        cleaned_keep.astype(np.uint8) * 255
    )
    if final_points is None:
        return []

    rx, ry, rw, rh = cv2.boundingRect(final_points)

    # Diagnostics intentionally expose whether processing left the candidate
    # visually disconnected. This does NOT create extra PNGs; it is useful
    # during the remaining validation folders.
    final_count, _, final_stats, _ = cv2.connectedComponentsWithStats(
        cleaned_keep.astype(np.uint8),
        connectivity=8,
    )
    final_component_areas = [
        int(final_stats[i, cv2.CC_STAT_AREA])
        for i in range(1, final_count)
        if int(final_stats[i, cv2.CC_STAT_AREA]) > 0
    ]

    cleanup_diagnostics = {
        "mode": "one_candidate_one_png_provenance",
        "components_before_cleanup": int(count - 1),
        "components_after_cleanup": int(max(0, final_count - 1)),
        "component_areas_after_cleanup": final_component_areas,
        "removed_tiny_components": int(removed_tiny),
        "removed_strip_residue_components": int(removed_strip_residue),
        "preserved_significant_components": int(preserved_components),
        "visually_disconnected_after_processing": bool(final_count - 1 > 1),
    }

    return [{
        "crop_bgr": crop_bgr,
        "keep_mask": cleaned_keep,
        "outer_roi_mask": export_data.get("outer_roi_mask"),
        "roi_x": rx,
        "roi_y": ry,
        "roi_width": rw,
        "roi_height": rh,
        "crop_global_x": export_data["crop_global_x"],
        "crop_global_y": export_data["crop_global_y"],
        "global_x": export_data["crop_global_x"] + rx,
        "global_y": export_data["crop_global_y"] + ry,
        "strip_diagnostics": strip_diagnostics,
        "cleanup_diagnostics": cleanup_diagnostics,
    }]


# ---------------------------------------------------------
# Dataset export
# ---------------------------------------------------------

def generate_dataset_zip(
    image_contents,
    target_count,
    condition,
    material,
    min_area=20,
    max_area=8000,
    target_size=320,
    export_brightness=10,
    dark_limit=9,
):
    """
    Generate the AI-ready ZIP.

    Detection/localization:
        BLRDetector auto-calibration.

    Final export:
        ImageJ-compatible extraction mask.

    Output:
        - 320x320 PNG
        - pure blue background
        - no object resize
        - original scanner RGB/BGR pixels preserved
        - very dark internal holes remain blue
    """

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

    (
        detection_brightness,
        detection_kernel,
        detector_labels,
        candidates,
    ) = _find_best_detection(
        bgr,
        target_count,
        min_area,
        max_area,
    )

    candidates.sort(
        key=lambda obj: (
            obj["y"],
            obj["x"],
        )
    )

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

        for candidate in candidates:
            export_data = _build_imagej_export_mask(
                bgr=bgr,
                detector_labels=detector_labels,
                candidate=candidate,
                brightness_min=export_brightness,
                dark_limit=dark_limit,
            )

            if export_data is None:
                skipped += 1
                continue

            single_piece_exports = _split_export_into_single_piece_regions(
                export_data,
                small_fragment_ratio=0.10,
                split_fragment_ratio=0.35,
                min_fragment_area_px=12,
            )

            if not single_piece_exports:
                skipped += 1
                continue

            for piece_export in single_piece_exports:
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

                # Same as the ImageJ macro:
                # oversized objects are not resized.
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

                canvas[:, :] = [
                    255,
                    0,
                    0,
                ]

                paste_x = (
                    target_size - width
                ) // 2

                paste_y = (
                    target_size - height
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

                success, png_buffer = cv2.imencode(
                    ".png",
                    canvas,
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
                            piece_export["global_x"],
                        "y":
                            piece_export["global_y"],
                        "width_px":
                            width,
                        "height_px":
                            height,
                        "detection_brightness":
                            detection_brightness,
                        "detection_kernel":
                            detection_kernel,
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

    zip_buffer.seek(0)

    diagnostics = {
        "detected":
            len(candidates),
        "exported":
            len(metadata_rows),
        "skipped":
            skipped,
        "brightness_min":
            detection_brightness,
        "kernel_size":
            detection_kernel,
        "export_brightness":
            export_brightness,
        "dark_limit":
            dark_limit,
    }

    return (
        zip_buffer.getvalue(),
        diagnostics,
    )
