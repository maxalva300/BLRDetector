(function () {
    "use strict";
    function nu() { return window.dash_clientside.no_update; }
    function trigger() {
        const t = window.dash_clientside.callback_context.triggered || [];
        return t.length ? t[0].prop_id.split(".")[0] : "";
    }
    const api = {
        selectObject: function (click, addMode) {
            const unchanged = () => Array(6).fill(nu());
            if (addMode || !click || !click.points || !click.points.length) return unchanged();
            const c = click.points[0].customdata;
            if (!Array.isArray(c) || c.length < 3) return unchanged();
            const state = String(c[0]), label = Number(c[1]), number = Number(c[2]);
            if (!["green", "yellow", "manual"].includes(state) || !Number.isFinite(label) || !Number.isFinite(number)) return unchanged();
            const selected = {state: state, label: label, visible_number: number};
            if (c.length >= 7) selected.bbox = {x: Number(c[3]), y: Number(c[4]), width: Number(c[5]), height: Number(c[6])};
            if (c.length >= 8 && c[7]) selected.repair_type = String(c[7]);
            if (c.length >= 9 && c[8] !== "") selected.source_label = Number(c[8]);
            if (c.length >= 10 && c[9]) selected.source_state = String(c[9]);
            let action = "Exclude", color = "warning", status = "Selected green object " + number + " · green → yellow";
            if (state === "yellow") {
                action = "Accept"; color = "success"; status = "Selected S" + number + " · yellow → green";
            } else if (state === "manual") {
                action = "Remove"; color = "danger";
                status = "Selected " + (selected.repair_type === "refined" ? "refined" : "added") + " object " + number + " · remove repair";
            }
            return [selected, false, action, color, {display: "inline-block"}, status];
        },
        rememberView: function (data, cache, addMode, saved) {
            const revision = (cache || {}).review_revision || "legacy-review";
            if (trigger() === "qc-cache-store") return [{revision: revision}, nu()];
            if (!data) return [nu(), nu()];
            const viewport = saved && saved.revision === revision ? Object.assign({}, saved) : {revision: revision};
            let changed = false;
            ["xaxis", "yaxis"].forEach(function (axis) {
                if (data[axis + ".autorange"] === true) {
                    delete viewport[axis]; changed = true; return;
                }
                let bounds = data[axis + ".range"];
                if (!bounds && axis + ".range[0]" in data && axis + ".range[1]" in data)
                    bounds = [data[axis + ".range[0]"], data[axis + ".range[1]"]];
                if (Array.isArray(bounds) && bounds.length === 2 && bounds.every(v => typeof v === "number" && Number.isFinite(v))) {
                    viewport[axis] = bounds.slice(); changed = true;
                }
            });
            const shapeEvent = Object.keys(data).some(k => k === "shapes" ? Array.isArray(data.shapes) && data.shapes.length > 0 : k.startsWith("shapes["));
            const drawing = addMode && shapeEvent ? Object.assign({}, data, {drawing_event: Date.now()}) : nu();
            return [changed ? viewport : nu(), drawing];
        },
        attachImage: function (figure, image, viewport) {
            if (!figure) return nu();
            if (!figure.layout) return figure;
            const f = Object.assign({}, figure), layout = Object.assign({}, figure.layout);
            if (layout.images) layout.images = layout.images.map(function (entry) {
                return entry.source === "__BLR_RGB__" ? Object.assign({}, entry, {source: image}) : entry;
            });
            const revision = (layout.meta || {}).blr_revision;
            if (viewport && revision && viewport.revision === revision) {
                ["xaxis", "yaxis"].forEach(function (axis) {
                    if (Array.isArray(viewport[axis])) layout[axis] = Object.assign({}, layout[axis] || {}, {range: viewport[axis].slice(), autorange: false});
                });
            }
            f.layout = layout;
            return f;
        },
        refineButton: function (selected, addMode) {
            return addMode || !selected || !selected.bbox ? [true, {display: "none"}] : [false, {display: "inline-block"}];
        },
        refineModal: function (open, cancel, apply, selected, isOpen) {
            const id = trigger();
            if (id === "refine-object-button") return selected && selected.bbox ? [true, selected] : [false, null];
            if (id === "refine-cancel-button") return [false, null];
            if (id === "refine-apply-button") return [false, nu()];
            return [isOpen, nu()];
        },
        refineControls: function (target, reset, cache) {
            if (!target && trigger() !== "refine-reset-button") return [nu(), nu(), ""];
            const manual = (cache || {}).manual_brightness_threshold;
            const baseline = manual === null || manual === undefined ? 10 : Number(manual);
            return [baseline, 6, "Batch baseline: brightness threshold " + baseline + "."];
        }
    };
    window.dash_clientside = Object.assign({}, window.dash_clientside, {blr_fast: api});
}());
