"""Reader for Blue Line Spectro+ Metadata.xls files."""

import math

import pandas as pd
import xlrd


def _clean_header(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _make_unique(names):
    counts = {}
    result = []
    for name in names:
        base = name or "column"
        count = counts.get(base, 0)
        result.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return result


def read_metadata(contents):
    """
    Parse the scanner's legacy .xls metadata bytes into a DataFrame.

    The Spectro+ file uses two header rows, e.g. ``size`` + ``width`` and
    ``area`` + ``pixel``. They are combined to produce names such as
    ``size.width``, ``size.height`` and ``area.pixel`` used by Geometry QC.
    """
    if not contents:
        raise ValueError("Metadata.xls is empty.")

    workbook = xlrd.open_workbook(file_contents=contents)
    if workbook.nsheets < 1:
        raise ValueError("Metadata.xls does not contain a worksheet.")

    sheet = workbook.sheet_by_index(0)
    if sheet.nrows < 3:
        raise ValueError("Metadata.xls does not contain enough rows.")

    parent_headers = [_clean_header(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
    child_headers = [_clean_header(sheet.cell_value(1, c)) for c in range(sheet.ncols)]

    headers = []
    for c, (parent, child) in enumerate(zip(parent_headers, child_headers)):
        if parent and child:
            if parent.lower() == child.lower():
                header = parent
            else:
                header = f"{parent}.{child}"
        elif parent:
            header = parent
        elif child:
            header = child
        else:
            header = f"column_{c}"
        headers.append(header)

    headers = _make_unique(headers)

    rows = []
    for r in range(2, sheet.nrows):
        row = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        if any(str(value).strip() for value in row):
            rows.append(row)

    return pd.DataFrame(rows, columns=headers)
