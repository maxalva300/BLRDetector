import os
import base64

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    ctx,
    no_update,
)
import dash_bootstrap_components as dbc

from utils.metadata_reader import read_metadata
from utils.geometry_qc import (
    build_geometry_qc_preview,
    create_manual_object_spec,
    generate_green_dataset_zip,
    prepare_geometry_qc_cache,
)


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP
    ],
)

app.title = "BLR Detector"
server = app.server

# Keep uploads in memory; no persistent filesystem is required.
server.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


CAROUSEL_SLIDES = [
    {
        "src": "/assets/bluelinemoduls.webp",
        "title": "BLR Detector",
        "subtitle": (
            "Fast, reviewable dataset preparation for sensor-based "
            "recycling images."
        ),
        "position": "center",
    },
    {
        "src": (
            "https://www.blueline-recycling.com/wp2019/"
            "wp-content/uploads/2021/09/"
            "inbetriebnahme-schauffler_bearb.jpg"
        ),
        "title": "From scanner batch to AI-ready objects",
        "subtitle": (
            "Upload RGB shelf data, review detections, and export "
            "consistent 320 × 320 object images."
        ),
        "position": "center",
    },
]


def _hero_slide(slide):
    return html.Div(
        className="blr-hero-slide",
        children=[
            html.Img(
                src=slide["src"],
                className="blr-hero-image",
                style={
                    "objectPosition":
                        slide.get(
                            "position",
                            "center",
                        )
                },
            ),
            html.Div(
                className="blr-hero-overlay",
                children=[
                    html.Div(
                        className="blr-hero-copy",
                        children=[
                            html.H1(
                                slide["title"],
                                className="blr-hero-title",
                            ),
                            html.P(
                                slide["subtitle"],
                                className="blr-hero-subtitle",
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------

UPLOAD_STYLE = {
    "width":
        "100%",
    "height":
        "90px",
    "lineHeight":
        "90px",
    "borderWidth":
        "1px",
    "borderStyle":
        "dashed",
    "borderRadius":
        "6px",
    "textAlign":
        "center",
    "cursor":
        "pointer",
}

FILE_OK_STYLE = {
    "marginTop":
        "8px",
    "fontSize":
        "14px",
    "fontWeight":
        "500",
}


# ---------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------

def _build_preview_summary(
    scanner_count,
    accepted_count,
    suspicious_count,
    diagnostics,
):
    """
    Compact image-first preview header.

    The preview should communicate the current dataset state without
    pushing the image down with cards and alerts.
    """

    difference = (
        accepted_count
        - scanner_count
    )

    difference_text = (
        f"+{difference}"
        if difference > 0
        else str(
            difference
        )
    )

    if difference == 0:
        status_badge = dbc.Badge(
            "Ready",
            color="success",
            pill=True,
            className="ms-2",
        )
    else:
        status_badge = dbc.Badge(
            "Review",
            color="warning",
            text_color="dark",
            pill=True,
            className="ms-2",
        )

    stats = html.Div(
        [
            html.Span(
                [
                    html.Span(
                        "Scanner ",
                        className="text-muted",
                    ),
                    html.Strong(
                        scanner_count
                    ),
                ],
                className="me-4",
            ),

            html.Span(
                [
                    html.Span(
                        "Green ",
                        className="text-muted",
                    ),
                    html.Strong(
                        accepted_count,
                        style={
                            "color":
                                "#16a34a"
                        },
                    ),
                ],
                className="me-4",
            ),

            html.Span(
                [
                    html.Span(
                        "Yellow ",
                        className="text-muted",
                    ),
                    html.Strong(
                        suspicious_count,
                        style={
                            "color":
                                "#d97706"
                        },
                    ),
                ],
                className="me-4",
            ),

            html.Span(
                [
                    html.Span(
                        "Δ ",
                        className="text-muted",
                    ),
                    html.Strong(
                        difference_text
                    ),
                ],
            ),

            status_badge,
        ],
        style={
            "fontSize":
                "16px",
            "whiteSpace":
                "nowrap",
        },
    )

    technical_details = html.Details(
        [
            html.Summary(
                "Technical details",
                style={
                    "cursor":
                        "pointer",
                    "color":
                        "#6c757d",
                    "fontSize":
                        "13px",
                    "userSelect":
                        "none",
                },
            ),

            html.Div(
                [
                    html.Small(
                        (
                            f"Brightness ≥ "
                            f"{diagnostics['brightness_min']} · "
                            f"Morphology "
                            f"{diagnostics['kernel_size']}×"
                            f"{diagnostics['kernel_size']} · "
                            f"Raw candidates "
                            f"{diagnostics['raw_candidate_count']} · "
                            f"Tested settings "
                            f"{diagnostics['evaluated_settings']} · "
                            f"Analysis "
                            f"{diagnostics['elapsed_seconds']:.2f} s"
                        ),
                        className="text-muted",
                    )
                ],
                className="pt-1",
            ),
        ],
        style={
            "textAlign":
                "right",
        },
    )

    return dbc.Row(
        [
            dbc.Col(
                stats,
                width="auto",
            ),
            dbc.Col(
                technical_details,
                className="ms-auto",
                width="auto",
            ),
        ],
        align="center",
        className="g-2 px-1 py-1",
    )


# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

app.layout = html.Div(
    className="blr-app-shell",
    children=[
        dcc.Interval(
            id="hero-interval",
            interval=5000,
            n_intervals=0,
        ),

        html.Div(
            className="blr-topbar",
            children=[
                html.Div(
                    className="blr-topbar-inner",
                    children=[
                        html.Img(
                            src="/assets/bluelinelogo.png",
                            className="blr-brand-logo",
                        ),
                        html.Div(
                            className="blr-topbar-title",
                            children=[
                                html.Strong("BLR Detector"),
                                html.Span(
                                    "AI Dataset Preparation",
                                    className="blr-topbar-subtitle",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        html.Main(
            className="blr-main",
            children=[
                html.Section(
                    className="blr-hero",
                    children=[
                        html.Div(
                            id="hero-carousel-content",
                            children=_hero_slide(
                                CAROUSEL_SLIDES[0]
                            ),
                        ),
                        html.Div(
                            id="hero-dots",
                            className="blr-hero-dots",
                        ),
                    ],
                ),

                html.Section(
                    className="blr-section-card",
                    children=[
                        html.Div(
                            className="blr-section-header",
                            children=[
                                html.Div(
                                    className="blr-step-number",
                                    children="1",
                                ),
                                html.Div(
                                    [
                                        html.H2(
                                            "Configure batch",
                                            className="blr-section-title",
                                        ),
                                        html.P(
                                            "Choose the sample condition and material class.",
                                            className="blr-section-copy",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dcc.Dropdown(
                            id="condition-dropdown",
                            options=[
                                {"label": "Treated", "value": "treated"},
                                {"label": "Not Treated", "value": "non_treated"},
                            ],
                            style={"display": "none"},
                        ),
                        dcc.Dropdown(
                            id="material-dropdown",
                            options=[
                                {"label": "Copper", "value": "copper"},
                                {"label": "Brass", "value": "brass"},
                                {"label": "Stainless Steel", "value": "stainless"},
                                {"label": "Grey Metal / Others", "value": "grey"},
                                {"label": "Coins", "value": "coins"},
                            ],
                            style={"display": "none"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Sample condition", className="blr-field-label"),
                                        html.Div(
                                            className="blr-pill-group",
                                            children=[
                                                dbc.Button("Treated", id="condition-treated-btn", className="blr-pill", color="light"),
                                                dbc.Button("Not Treated", id="condition-nt-btn", className="blr-pill", color="light"),
                                            ],
                                        ),
                                    ],
                                    lg=5,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Material class", className="blr-field-label"),
                                        html.Div(
                                            className="blr-pill-group",
                                            children=[
                                                dbc.Button("Copper", id="material-copper-btn", className="blr-pill", color="light"),
                                                dbc.Button("Brass", id="material-brass-btn", className="blr-pill", color="light"),
                                                dbc.Button("Stainless Steel", id="material-stainless-btn", className="blr-pill", color="light"),
                                                dbc.Button("Grey Metal", id="material-grey-btn", className="blr-pill", color="light"),
                                                dbc.Button("Coins", id="material-coins-btn", className="blr-pill", color="light"),
                                            ],
                                        ),
                                    ],
                                    lg=7,
                                ),
                            ],
                            className="g-4",
                        ),
                    ],
                ),

                html.Section(
                    className="blr-section-card",
                    children=[
                        html.Div(
                            className="blr-section-header",
                            children=[
                                html.Div(className="blr-step-number", children="2"),
                                html.Div(
                                    [
                                        html.H2("Load scanner batch", className="blr-section-title"),
                                        html.P("Upload the RGB shelf image and its Metadata.xls file.", className="blr-section-copy"),
                                    ]
                                ),
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Upload(
                                            id="upload-image",
                                            className="blr-upload-card",
                                            children=html.Div(
                                                className="blr-upload-inner",
                                                children=[
                                                    html.Div("RGB", className="blr-rgb-icon"),
                                                    html.Div([
                                                        html.Div("RGB Shelf Image", className="blr-upload-title"),
                                                        html.Div("Drop RGB_shelf.png here or click to browse", className="blr-upload-copy"),
                                                    ]),
                                                ],
                                            ),
                                            multiple=False,
                                            accept=".png,.jpg,.jpeg",
                                        ),
                                        html.Div(id="image-upload-status", className="blr-file-status"),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Upload(
                                            id="upload-metadata",
                                            className="blr-upload-card",
                                            children=html.Div(
                                                className="blr-upload-inner",
                                                children=[
                                                    html.Img(src="/assets/excel-icon.png", className="blr-upload-icon-img"),
                                                    html.Div([
                                                        html.Div("Batch Metadata", className="blr-upload-title"),
                                                        html.Div("Drop Metadata.xls here or click to browse", className="blr-upload-copy"),
                                                    ]),
                                                ],
                                            ),
                                            multiple=False,
                                            accept=".xls",
                                        ),
                                        html.Div(id="metadata-upload-status", className="blr-file-status"),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-4",
                        ),
                        html.Div(
                            className="blr-primary-action-row",
                            children=[
                                dbc.Button("Analyze Batch", id="analyze-button", className="blr-primary-btn", color="primary", size="lg")
                            ],
                        ),
                    ],
                ),

                html.Section(
                    className="blr-section-card",
                    children=[
                        html.Div(
                            className="blr-section-header blr-section-header-compact",
                            children=[
                                html.Div(className="blr-step-number", children="3"),
                                html.Div([
                                    html.H2("Review & export", className="blr-section-title"),
                                    html.P("Review only when needed, then generate the final dataset.", className="blr-section-copy"),
                                ]),
                            ],
                        ),
                        html.Div(id="batch-summary"),
                        html.Div(
                            className="blr-action-row",
                            children=[
                                html.Div(
                                    [dbc.Button("Review Objects", id="preview-button", className="blr-secondary-btn", color="secondary", outline=True)],
                                    id="preview-button-container",
                                    style={"display": "none"},
                                ),
                                html.Div(
                                    [dbc.Button("Generate Dataset", id="generate-button", className="blr-primary-btn", color="success")],
                                    id="generate-button-container",
                                    style={"display": "none"},
                                ),
                            ],
                        ),
                        dbc.Collapse(
                            id="preview-modal",
                            is_open=False,
                            children=[
                                html.Div(
                                    className="blr-inline-preview",
                                    children=[
                                        html.Div(
                                            className="blr-preview-toolbar",
                                            children=[
                                                html.H3("Batch Preview", className="blr-preview-title"),
                                                html.Div(
                                                    className="blr-preview-tools",
                                                    children=[
                                                        dbc.Button("i", id="preview-info-button", color="primary", outline=True, size="sm", className="rounded-circle blr-info-btn"),
                                                        dbc.Button("Hide", id="close-preview-button", color="secondary", outline=True, size="sm"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dbc.Popover(
                                            [
                                                dbc.PopoverHeader("Preview help"),
                                                dbc.PopoverBody([
                                                    html.Div([html.Strong("Green: "), "included in Generate Dataset."], className="mb-2"),
                                                    html.Div([html.Strong("Yellow: "), "excluded unless you accept it manually."], className="mb-2"),
                                                    html.Div([html.Strong("Manual review: "), "yellow → green with Accept; green → yellow with Exclude."], className="mb-2"),
                                                    html.Div([html.Strong("Add Object: "), "draw one box around one complete missing or fragmented object."], className="mb-2"),
                                                    html.Div([html.Strong("Generate Dataset: "), "exports only the objects that are green."]),
                                                ]),
                                            ],
                                            id="preview-info-popover",
                                            target="preview-info-button",
                                            trigger="legacy",
                                            placement="bottom-end",
                                        ),
                                        html.Div(id="modal-preview-content", className="blr-preview-summary"),
                                        dcc.Graph(
                                            id="preview-graph",
                                            figure={},
                                            config={
                                                "scrollZoom": True,
                                                "responsive": True,
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                                "modeBarButtonsToAdd": ["pan2d"],
                                                "doubleClick": "reset",
                                            },
                                            style={"display": "none", "width": "100%", "height": "70vh"},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        id="review-selection-status",
                                                        children="Optional review: click an object only if you want to change its status.",
                                                        className="blr-review-status",
                                                    ),
                                                    md=8,
                                                ),
                                                dbc.Col(
                                                    html.Div([
                                                        dbc.Button("Add Object", id="add-object-button", color="primary", outline=True, size="sm", className="me-2"),
                                                        dbc.Button("Review action", id="review-action-button", color="secondary", disabled=True, size="sm", style={"display": "none"}),
                                                    ]),
                                                    md=4,
                                                    className="text-end",
                                                ),
                                            ],
                                            className="g-2 align-items-center blr-review-bar",
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dcc.Download(id="download-dataset"),
                        html.Div(id="generation-status", className="mt-3"),
                    ],
                ),

                dcc.Store(id="manual-accepted-store", data=[]),
                dcc.Store(id="manual-excluded-store", data=[]),
                dcc.Store(id="selected-object-store", data=None),
                dcc.Store(id="manual-objects-store", data=[]),
                dcc.Store(id="add-object-mode-store", data=False),
                dcc.Store(id="qc-cache-store", data=None),

                html.Footer(
                    className="blr-footer",
                    children="BLR Detector · Blue Line Recycling dataset preparation",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------

@app.callback(
    Output("hero-carousel-content", "children"),
    Output("hero-dots", "children"),
    Input("hero-interval", "n_intervals"),
)
def update_hero(n_intervals):
    index = n_intervals % len(CAROUSEL_SLIDES)
    dots = [
        html.Span(
            className=("blr-hero-dot active" if idx == index else "blr-hero-dot")
        )
        for idx in range(len(CAROUSEL_SLIDES))
    ]
    return _hero_slide(CAROUSEL_SLIDES[index]), dots


@app.callback(
    Output("condition-dropdown", "value"),
    Output("condition-treated-btn", "className"),
    Output("condition-nt-btn", "className"),
    Input("condition-treated-btn", "n_clicks"),
    Input("condition-nt-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_condition(treated_clicks, nt_clicks):
    selected = "treated" if ctx.triggered_id == "condition-treated-btn" else "non_treated"
    return (
        selected,
        "blr-pill active" if selected == "treated" else "blr-pill",
        "blr-pill active" if selected == "non_treated" else "blr-pill",
    )


@app.callback(
    Output("material-dropdown", "value"),
    Output("material-copper-btn", "className"),
    Output("material-brass-btn", "className"),
    Output("material-stainless-btn", "className"),
    Output("material-grey-btn", "className"),
    Output("material-coins-btn", "className"),
    Input("material-copper-btn", "n_clicks"),
    Input("material-brass-btn", "n_clicks"),
    Input("material-stainless-btn", "n_clicks"),
    Input("material-grey-btn", "n_clicks"),
    Input("material-coins-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_material(copper_clicks, brass_clicks, stainless_clicks, grey_clicks, coins_clicks):
    mapping = {
        "material-copper-btn": "copper",
        "material-brass-btn": "brass",
        "material-stainless-btn": "stainless",
        "material-grey-btn": "grey",
        "material-coins-btn": "coins",
    }
    selected = mapping.get(ctx.triggered_id)
    values = ["copper", "brass", "stainless", "grey", "coins"]
    classes = ["blr-pill active" if value == selected else "blr-pill" for value in values]
    return (selected, *classes)


# ---------------------------------------------------------
# Upload indicators
# ---------------------------------------------------------

@app.callback(
    Output("image-upload-status", "children"),
    Output("upload-image", "className"),
    Input("upload-image", "filename"),
)
def show_image_filename(filename):
    if not filename:
        return "", "blr-upload-card"
    return (
        html.Div([
            html.Span("✓", className="blr-loaded-check"),
            html.Span(filename, className="blr-loaded-name"),
            html.Span("Loaded", className="blr-loaded-label"),
        ], className="blr-loaded-row"),
        "blr-upload-card loaded",
    )


@app.callback(
    Output("metadata-upload-status", "children"),
    Output("upload-metadata", "className"),
    Input("upload-metadata", "filename"),
)
def show_metadata_filename(filename):
    if not filename:
        return "", "blr-upload-card"
    return (
        html.Div([
            html.Span("✓", className="blr-loaded-check"),
            html.Span(filename, className="blr-loaded-name"),
            html.Span("Loaded", className="blr-loaded-label"),
        ], className="blr-loaded-row"),
        "blr-upload-card loaded",
    )


# ---------------------------------------------------------
# Analyze batch
# ---------------------------------------------------------

@app.callback(
    Output("batch-summary", "children"),
    Output("preview-button-container", "style"),
    Output("generate-button-container", "style"),
    Output("manual-accepted-store", "data"),
    Output("manual-excluded-store", "data"),
    Output("selected-object-store", "data"),
    Output("manual-objects-store", "data"),
    Output("add-object-mode-store", "data"),
    Output("qc-cache-store", "data"),

    Input("analyze-button", "n_clicks"),

    State("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-metadata", "contents"),
    State("upload-metadata", "filename"),
    State("condition-dropdown", "value"),
    State("material-dropdown", "value"),

    prevent_initial_call=True,
)
def analyze_batch(
    n_clicks,
    image_contents,
    image_filename,
    metadata_contents,
    metadata_filename,
    condition,
    material,
):
    hidden = {"display": "none"}
    visible = {"display": "inline-block"}

    if image_contents is None or metadata_contents is None:
        return (
            dbc.Alert(
                "Please upload both the RGB shelf image and Metadata.xls.",
                color="warning",
            ),
            hidden,
            hidden,
            [], [], None, [], False, None,
        )

    if condition is None or material is None:
        return (
            dbc.Alert(
                "Please select the sample condition and material class.",
                color="warning",
            ),
            hidden,
            hidden,
            [], [], None, [], False, None,
        )

    try:
        _, metadata_string = metadata_contents.split(",", 1)
        metadata_bytes = base64.b64decode(metadata_string)
        df = read_metadata(metadata_bytes)

        # V0.54: do the expensive 40-setting search ONCE here.
        # The dcc.Store receives only compact JSON-safe parameters/labels,
        # never the image or full-size NumPy label matrices.
        qc_cache = prepare_geometry_qc_cache(
            image_contents,
            df,
        )

    except Exception as e:
        return (
            dbc.Alert(f"Analysis error: {e}", color="danger"),
            hidden,
            hidden,
            [], [], None, [], False, None,
        )

    object_count = len(df)

    condition_names = {
        "treated": "Treated",
        "non_treated": "Non-treated",
    }
    material_names = {
        "copper": "Copper",
        "brass": "Brass",
        "stainless": "Stainless Steel",
        "grey": "Grey Metal / Others",
        "coins": "Coins",
    }

    condition_label = condition_names.get(condition, condition)
    material_label = material_names.get(material, material)

    diagnostics = qc_cache.get("diagnostics", {})
    auto_green = diagnostics.get("automatic_accepted_count", 0)
    auto_yellow = diagnostics.get("automatic_suspicious_count", 0)

    summary = dbc.Card(
        [
            dbc.CardHeader(html.H5("Batch Summary", className="mb-0")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Small("Sample condition", className="text-muted"),
                                    html.H5(condition_label),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Material class", className="text-muted"),
                                    html.H5(material_label),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Scanner objects", className="text-muted"),
                                    html.H3(str(object_count), className="mb-0"),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Status", className="text-muted"),
                                    html.H5(
                                        "Analyzed · Ready for review",
                                        style={"color": "green"},
                                    ),
                                ],
                                md=3,
                            ),
                        ]
                    ),
                    html.Hr(),
                    html.Small(
                        f"Automatic result: {auto_green} green · {auto_yellow} yellow",
                        className="text-muted d-block",
                    ),
                    html.Small(
                        f"Image: {image_filename}",
                        className="text-muted d-block",
                    ),
                    html.Small(
                        f"Metadata: {metadata_filename}",
                        className="text-muted d-block",
                    ),
                ]
            ),
        ]
    )

    # Analyze also resets all manual review state.
    return (
        summary,
        visible,
        visible,
        [],
        [],
        None,
        [],
        False,
        qc_cache,
    )


# ---------------------------------------------------------
# Preview + optional manual review
# ---------------------------------------------------------

@app.callback(
    Output(
        "preview-modal",
        "is_open"
    ),
    Output(
        "modal-preview-content",
        "children"
    ),
    Output(
        "preview-graph",
        "figure"
    ),
    Output(
        "preview-graph",
        "style"
    ),
    Output(
        "manual-accepted-store",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "manual-excluded-store",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "selected-object-store",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "manual-objects-store",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "add-object-mode-store",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "review-action-button",
        "disabled"
    ),
    Output(
        "review-action-button",
        "children"
    ),
    Output(
        "review-action-button",
        "color"
    ),
    Output(
        "review-action-button",
        "style"
    ),
    Output(
        "add-object-button",
        "children"
    ),
    Output(
        "add-object-button",
        "color"
    ),
    Output(
        "review-selection-status",
        "children"
    ),

    Input(
        "preview-button",
        "n_clicks"
    ),
    Input(
        "close-preview-button",
        "n_clicks"
    ),
    Input(
        "preview-graph",
        "clickData"
    ),
    Input(
        "preview-graph",
        "relayoutData"
    ),
    Input(
        "review-action-button",
        "n_clicks"
    ),
    Input(
        "add-object-button",
        "n_clicks"
    ),

    State(
        "preview-modal",
        "is_open"
    ),
    State(
        "upload-image",
        "contents"
    ),
    State(
        "upload-metadata",
        "contents"
    ),
    State(
        "manual-accepted-store",
        "data"
    ),
    State(
        "manual-excluded-store",
        "data"
    ),
    State(
        "selected-object-store",
        "data"
    ),
    State(
        "manual-objects-store",
        "data"
    ),
    State(
        "add-object-mode-store",
        "data"
    ),
    State(
        "preview-graph",
        "figure"
    ),
    State(
        "qc-cache-store",
        "data"
    ),

    prevent_initial_call=True,
)
def review_preview(
    preview_clicks,
    close_clicks,
    click_data,
    relayout_data,
    action_clicks,
    add_clicks,
    is_open,
    image_contents,
    metadata_contents,
    manual_accepted_labels,
    manual_excluded_labels,
    selected_object,
    manual_objects,
    add_object_mode,
    current_figure,
    qc_cache,
):
    triggered = ctx.triggered_id

    triggered_prop = (
        ctx.triggered[
            0
        ][
            "prop_id"
        ]
        if ctx.triggered
        else ""
    )

    manual_accepted_labels = (
        manual_accepted_labels
        or []
    )

    manual_excluded_labels = (
        manual_excluded_labels
        or []
    )

    manual_objects = (
        manual_objects
        or []
    )

    add_object_mode = bool(
        add_object_mode
    )

    graph_style = {
        "display":
            "block",
        "width":
            "100%",
        "height":
            "68vh",
    }

    hidden_graph_style = {
        "display":
            "none",
        "width":
            "100%",
        "height":
            "68vh",
    }

    hidden_button_style = {
        "display":
            "none"
    }

    visible_button_style = {
        "display":
            "inline-block"
    }

    default_status = (
        "Optional review: click an object only if you want "
        "to change its status."
    )

    add_status = (
        "Add Object active · draw one rectangle around one complete "
        "missing or fragmented object."
    )

    def load_metadata():
        _, metadata_string = (
            metadata_contents.split(
                ",",
                1,
            )
        )

        metadata_bytes = (
            base64.b64decode(
                metadata_string
            )
        )

        return read_metadata(
            metadata_bytes
        )

    def build_current_preview(
        accepted_labels,
        excluded_labels,
        current_manual_objects,
    ):
        df = load_metadata()

        (
            figure,
            accepted,
            suspicious,
            diagnostics,
        ) = build_geometry_qc_preview(
            image_contents,
            df,
            manual_accepted_labels=
                accepted_labels,
            manual_excluded_labels=
                excluded_labels,
            manual_objects=
                current_manual_objects,
            qc_cache=
                qc_cache,
        )

        summary = (
            _build_preview_summary(
                scanner_count=
                    len(
                        df
                    ),
                accepted_count=
                    len(
                        accepted
                    ),
                suspicious_count=
                    len(
                        suspicious
                    ),
                diagnostics=
                    diagnostics,
            )
        )

        return (
            figure,
            summary,
        )

    def figure_for_add_mode(
        figure,
        active,
    ):
        updated = dict(
            figure
            or {}
        )

        layout = dict(
            updated.get(
                "layout",
                {}
            )
        )

        layout[
            "dragmode"
        ] = (
            "drawrect"
            if active
            else "zoom"
        )

        layout[
            "shapes"
        ] = []

        if active:
            layout[
                "newshape"
            ] = {
                "line":
                    {
                        "color":
                            "#0d6efd",
                        "width":
                            2,
                    },
                "fillcolor":
                    "rgba(13,110,253,0.08)",
            }

        updated[
            "layout"
        ] = layout

        return updated

    def extract_rectangle(
        data,
    ):
        if not data:
            return None

        # Plotly may return the complete shape list.
        shapes = data.get(
            "shapes"
        )

        if (
            isinstance(
                shapes,
                list,
            )
            and shapes
        ):
            shape = shapes[
                -1
            ]

            required = [
                "x0",
                "x1",
                "y0",
                "y1",
            ]

            if all(
                key in shape
                for key
                in required
            ):
                return {
                    key:
                        shape[
                            key
                        ]
                    for key
                    in required
                }

        # Or it may return flattened relayout keys.
        values = {}

        for key in [
            "x0",
            "x1",
            "y0",
            "y1",
        ]:
            suffix = (
                f".{key}"
            )

            matching = [
                value
                for field, value
                in data.items()
                if (
                    field.startswith(
                        "shapes["
                    )
                    and field.endswith(
                        suffix
                    )
                )
            ]

            if matching:
                values[
                    key
                ] = matching[
                    -1
                ]

        if len(
            values
        ) == 4:
            return values

        return None

    if triggered == "close-preview-button":
        return (
            False,
            no_update,
            no_update,
            no_update,
            manual_accepted_labels,
            manual_excluded_labels,
            None,
            manual_objects,
            False,
            True,
            "Review action",
            "secondary",
            hidden_button_style,
            "Add Object",
            "primary",
            default_status,
        )

    # -----------------------------------------------------
    # Add Object button: enter/cancel rectangle mode
    # -----------------------------------------------------

    if triggered == "add-object-button":

        if (
            image_contents is None
            or metadata_contents is None
        ):
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                selected_object,
                manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                "Image or Metadata is missing.",
            )

        if add_object_mode:
            # Cancel without changing any object.
            try:
                (
                    figure,
                    summary,
                ) = build_current_preview(
                    manual_accepted_labels,
                    manual_excluded_labels,
                    manual_objects,
                )

                return (
                    True,
                    summary,
                    figure,
                    graph_style,
                    manual_accepted_labels,
                    manual_excluded_labels,
                    None,
                    manual_objects,
                    False,
                    True,
                    "Review action",
                    "secondary",
                    hidden_button_style,
                    "Add Object",
                    "primary",
                    default_status,
                )

            except Exception as e:
                return (
                    True,
                    dbc.Alert(
                        f"Preview error: {e}",
                        color="danger",
                    ),
                    no_update,
                    no_update,
                    manual_accepted_labels,
                    manual_excluded_labels,
                    selected_object,
                    manual_objects,
                    False,
                    True,
                    "Review action",
                    "secondary",
                    hidden_button_style,
                    "Add Object",
                    "primary",
                    no_update,
                )

        # Enter rectangle-drawing mode.
        if current_figure:
            add_figure = (
                figure_for_add_mode(
                    current_figure,
                    True,
                )
            )
        else:
            try:
                (
                    fresh_figure,
                    summary,
                ) = build_current_preview(
                    manual_accepted_labels,
                    manual_excluded_labels,
                    manual_objects,
                )

                add_figure = (
                    figure_for_add_mode(
                        fresh_figure,
                        True,
                    )
                )

            except Exception as e:
                return (
                    is_open,
                    dbc.Alert(
                        f"Preview error: {e}",
                        color="danger",
                    ),
                    no_update,
                    no_update,
                    manual_accepted_labels,
                    manual_excluded_labels,
                    selected_object,
                    manual_objects,
                    False,
                    True,
                    "Review action",
                    "secondary",
                    hidden_button_style,
                    "Add Object",
                    "primary",
                    no_update,
                )

        return (
            True,
            no_update,
            add_figure,
            graph_style,
            manual_accepted_labels,
            manual_excluded_labels,
            None,
            manual_objects,
            True,
            True,
            "Review action",
            "secondary",
            hidden_button_style,
            "Cancel Add",
            "danger",
            html.Span(
                add_status,
                style={
                    "color":
                        "#0d6efd"
                },
            ),
        )

    # -----------------------------------------------------
    # Rectangle drawn while Add Object is active
    # -----------------------------------------------------

    if (
        triggered_prop
        == "preview-graph.relayoutData"
        and add_object_mode
    ):
        rectangle = (
            extract_rectangle(
                relayout_data
            )
        )

        # Ignore normal zoom/pan relayout events.
        if rectangle is None:
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                selected_object,
                manual_objects,
                True,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Cancel Add",
                "danger",
                no_update,
            )

        try:
            next_manual_id = (
                max(
                    [
                        int(
                            item.get(
                                "manual_id",
                                0,
                            )
                        )
                        for item
                        in manual_objects
                    ],
                    default=0,
                )
                + 1
            )

            manual_object = (
                create_manual_object_spec(
                    image_contents=
                        image_contents,
                    roi=
                        rectangle,
                    manual_id=
                        next_manual_id,
                )
            )

            updated_manual_objects = list(
                manual_objects
            )

            updated_manual_objects.append(
                manual_object
            )

            (
                figure,
                summary,
            ) = build_current_preview(
                manual_accepted_labels,
                manual_excluded_labels,
                updated_manual_objects,
            )

            return (
                True,
                summary,
                figure,
                graph_style,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                updated_manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                html.Span(
                    (
                        "Added as one green object · overlapping "
                        "automatic fragments were replaced."
                    ),
                    style={
                        "color":
                            "#16a34a"
                    },
                ),
            )

        except Exception as e:
            # Keep Add mode active so the user can immediately retry.
            retry_figure = (
                figure_for_add_mode(
                    current_figure,
                    True,
                )
                if current_figure
                else no_update
            )

            return (
                True,
                no_update,
                retry_figure,
                graph_style,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                manual_objects,
                True,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Cancel Add",
                "danger",
                html.Span(
                    str(
                        e
                    ),
                    style={
                        "color":
                            "#dc3545"
                    },
                ),
            )

    # -----------------------------------------------------
    # Object click review
    # -----------------------------------------------------

    if (
        triggered_prop
        == "preview-graph.clickData"
    ):
        if (
            add_object_mode
            or not click_data
        ):
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                no_update,
                manual_objects,
                add_object_mode,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        points = click_data.get(
            "points",
            []
        )

        if not points:
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                no_update,
                manual_objects,
                False,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        customdata = points[
            0
        ].get(
            "customdata"
        )

        if (
            not isinstance(
                customdata,
                (
                    list,
                    tuple,
                ),
            )
            or len(
                customdata
            ) < 3
        ):
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                no_update,
                manual_objects,
                False,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        state = str(
            customdata[
                0
            ]
        )

        object_identifier = int(
            customdata[
                1
            ]
        )

        visible_number = int(
            customdata[
                2
            ]
        )

        selected = {
            "state":
                state,
            "label":
                object_identifier,
            "visible_number":
                visible_number,
        }

        if state == "yellow":
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                selected,
                manual_objects,
                False,
                False,
                "Accept",
                "success",
                visible_button_style,
                "Add Object",
                "primary",
                html.Span(
                    [
                        "Selected ",
                        html.Strong(
                            f"S{visible_number}"
                        ),
                        " · yellow → green",
                    ]
                ),
            )

        if state == "manual":
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                selected,
                manual_objects,
                False,
                False,
                "Remove",
                "danger",
                visible_button_style,
                "Add Object",
                "primary",
                html.Span(
                    [
                        "Selected added object ",
                        html.Strong(
                            str(
                                visible_number
                            )
                        ),
                        " · remove repair",
                    ]
                ),
            )

        return (
            is_open,
            no_update,
            no_update,
            no_update,
            manual_accepted_labels,
            manual_excluded_labels,
            selected,
            manual_objects,
            False,
            False,
            "Exclude",
            "warning",
            visible_button_style,
            "Add Object",
            "primary",
            html.Span(
                [
                    "Selected green object ",
                    html.Strong(
                        str(
                            visible_number
                        )
                    ),
                    " · green → yellow",
                ]
            ),
        )

    # -----------------------------------------------------
    # Accept / Exclude / Remove
    # -----------------------------------------------------

    if triggered == "review-action-button":

        if (
            not selected_object
            or selected_object.get(
                "label"
            )
            is None
        ):
            return (
                is_open,
                no_update,
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                default_status,
            )

        selected_label = int(
            selected_object[
                "label"
            ]
        )

        selected_state = str(
            selected_object.get(
                "state",
                ""
            )
        )

        updated_accepted = list(
            manual_accepted_labels
        )

        updated_excluded = list(
            manual_excluded_labels
        )

        updated_manual_objects = list(
            manual_objects
        )

        if selected_state == "yellow":
            if (
                selected_label
                not in updated_accepted
            ):
                updated_accepted.append(
                    selected_label
                )

            updated_excluded = [
                label
                for label
                in updated_excluded
                if int(
                    label
                )
                != selected_label
            ]

            action_message = (
                "Accepted · object is now green and will be exported."
            )

            action_color = "#16a34a"

        elif selected_state == "manual":
            updated_manual_objects = [
                item
                for item
                in updated_manual_objects
                if int(
                    item.get(
                        "manual_id",
                        -1,
                    )
                )
                != selected_label
            ]

            action_message = (
                "Added object removed · original automatic "
                "detections are visible again."
            )

            action_color = "#6c757d"

        else:
            if (
                selected_label
                not in updated_excluded
            ):
                updated_excluded.append(
                    selected_label
                )

            updated_accepted = [
                label
                for label
                in updated_accepted
                if int(
                    label
                )
                != selected_label
            ]

            action_message = (
                "Excluded · object is now yellow and will not be exported."
            )

            action_color = "#d97706"

        try:
            (
                figure,
                summary,
            ) = build_current_preview(
                updated_accepted,
                updated_excluded,
                updated_manual_objects,
            )

            return (
                True,
                summary,
                figure,
                graph_style,
                updated_accepted,
                updated_excluded,
                None,
                updated_manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                html.Span(
                    action_message,
                    style={
                        "color":
                            action_color
                    },
                ),
            )

        except Exception as e:
            return (
                True,
                dbc.Alert(
                    f"Preview error: {e}",
                    color="danger",
                ),
                no_update,
                no_update,
                manual_accepted_labels,
                manual_excluded_labels,
                selected_object,
                manual_objects,
                False,
                False,
                no_update,
                no_update,
                visible_button_style,
                "Add Object",
                "primary",
                no_update,
            )

    # -----------------------------------------------------
    # Open preview
    # -----------------------------------------------------

    if triggered == "preview-button":

        if (
            image_contents is None
            or metadata_contents is None
        ):
            return (
                False,
                "",
                {},
                hidden_graph_style,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                "",
            )

        try:
            (
                figure,
                summary,
            ) = build_current_preview(
                manual_accepted_labels,
                manual_excluded_labels,
                manual_objects,
            )

            return (
                True,
                summary,
                figure,
                graph_style,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                default_status,
            )

        except Exception as e:
            return (
                True,
                dbc.Alert(
                    f"Preview error: {e}",
                    color="danger",
                ),
                {},
                hidden_graph_style,
                manual_accepted_labels,
                manual_excluded_labels,
                None,
                manual_objects,
                False,
                True,
                "Review action",
                "secondary",
                hidden_button_style,
                "Add Object",
                "primary",
                "",
            )

    return (
        is_open,
        no_update,
        no_update,
        no_update,
        manual_accepted_labels,
        manual_excluded_labels,
        selected_object,
        manual_objects,
        add_object_mode,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    )

# ---------------------------------------------------------
# Generate dataset from CURRENT GREEN SET
# ---------------------------------------------------------

@app.callback(
    Output(
        "download-dataset",
        "data"
    ),
    Output(
        "generation-status",
        "children"
    ),

    Input(
        "generate-button",
        "n_clicks"
    ),

    State(
        "upload-image",
        "contents"
    ),
    State(
        "upload-metadata",
        "contents"
    ),
    State(
        "condition-dropdown",
        "value"
    ),
    State(
        "material-dropdown",
        "value"
    ),
    State(
        "manual-accepted-store",
        "data"
    ),
    State(
        "manual-excluded-store",
        "data"
    ),
    State(
        "manual-objects-store",
        "data"
    ),
    State(
        "qc-cache-store",
        "data"
    ),

    prevent_initial_call=True,
)
def generate_dataset(
    n_clicks,
    image_contents,
    metadata_contents,
    condition,
    material,
    manual_accepted_labels,
    manual_excluded_labels,
    manual_objects,
    qc_cache,
):
    try:
        _, metadata_string = (
            metadata_contents.split(
                ",",
                1,
            )
        )

        metadata_bytes = (
            base64.b64decode(
                metadata_string
            )
        )

        df = read_metadata(
            metadata_bytes
        )

        (
            zip_bytes,
            diagnostics,
        ) = generate_green_dataset_zip(
            image_contents=
                image_contents,
            metadata_df=
                df,
            manual_accepted_labels=
                manual_accepted_labels
                or [],
            manual_excluded_labels=
                manual_excluded_labels
                or [],
            manual_objects=
                manual_objects
                or [],
            condition=
                condition,
            material=
                material,
            qc_cache=
                qc_cache,
        )

        scanner_count = (
            diagnostics[
                "scanner_count"
            ]
        )

        green_count = (
            diagnostics[
                "green_count"
            ]
        )

        yellow_remaining = (
            diagnostics[
                "yellow_remaining"
            ]
        )

        exported_count = (
            diagnostics[
                "exported"
            ]
        )

        skipped_count = (
            diagnostics[
                "skipped"
            ]
        )

        export_ok = (
            exported_count
            == green_count
            and skipped_count
            == 0
        )

        qc_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Scanner objects",
                                    className=
                                        "text-muted",
                                ),
                                html.H3(
                                    scanner_count,
                                    className=
                                        "mb-0",
                                ),
                            ]
                        )
                    ),
                    md=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Green objects",
                                    className=
                                        "text-muted",
                                ),
                                html.H3(
                                    green_count,
                                    className=
                                        "mb-0",
                                    style={
                                        "color":
                                            "#16a34a"
                                    },
                                ),
                            ]
                        )
                    ),
                    md=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Exported",
                                    className=
                                        "text-muted",
                                ),
                                html.H3(
                                    exported_count,
                                    className=
                                        "mb-0",
                                ),
                            ]
                        )
                    ),
                    md=3,
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Small(
                                    "Skipped",
                                    className=
                                        "text-muted",
                                ),
                                html.H3(
                                    skipped_count,
                                    className=
                                        "mb-0",
                                ),
                            ]
                        )
                    ),
                    md=3,
                ),
            ],
            className=
                "g-2 mb-3",
        )

        if export_ok:

            difference = (
                green_count
                - scanner_count
            )

            if difference == 0:
                main_message = (
                    "All current green objects were exported successfully."
                )
            else:
                sign = (
                    "+"
                    if difference > 0
                    else ""
                )

                main_message = (
                    "All current green objects were exported successfully. "
                    f"The reviewed green set differs from the scanner count "
                    f"by {sign}{difference}."
                )

            validation_alert = dbc.Alert(
                [
                    html.Strong(
                        "✓ Dataset generated from green set"
                    ),
                    html.Div(
                        main_message
                    ),
                    html.Div(
                        (
                            f"{yellow_remaining} yellow object(s) "
                            "remained excluded from the ZIP."
                        ),
                        className=
                            "mt-1",
                    ),
                ],
                color=
                    "success",
                className=
                    "mb-0",
            )

        else:
            validation_alert = dbc.Alert(
                [
                    html.Strong(
                        "⚠ Export review required"
                    ),
                    html.Div(
                        (
                            f"Green objects: {green_count} | "
                            f"Exported: {exported_count} | "
                            f"Skipped: {skipped_count}"
                        ),
                    ),
                    html.Div(
                        (
                            f"{yellow_remaining} yellow object(s) "
                            "remained excluded."
                        ),
                        className=
                            "mt-1",
                    ),
                ],
                color=
                    "warning",
                className=
                    "mb-0",
            )

        status = html.Div(
            [
                html.H5(
                    "Dataset Validation",
                    className=
                        "mb-3",
                ),
                qc_cards,
                validation_alert,
            ]
        )

        zip_filename = (
            f"BLR_"
            f"{material}_"
            f"{condition}.zip"
        )

        return (
            dcc.send_bytes(
                zip_bytes,
                zip_filename,
            ),
            status,
        )

    except Exception as e:
        return (
            None,
            dbc.Alert(
                (
                    f"Dataset generation error: {e}"
                ),
                color=
                    "danger",
            ),
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
    )
