from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree.ElementTree import Element, SubElement, tostring
from zipfile import ZIP_DEFLATED, ZipFile


_NATURAL_RE = re.compile(r"(\d+)")
_INVALID_SHEET_CHARS_RE = re.compile(r"[\[\]\*\?/\\:]")
_XLSX_GEOMETRIC_FRACTION_COLUMNS = frozenset(
    {
        "radial_inner_fraction",
        "radial_outer_fraction",
        "radial_center_proj_fraction",
        "radial_center_caps_fraction",
        "radial_center_surfw_fraction",
        "inner_radius_fraction",
        "outer_radius_fraction",
    }
)
_XLSX_NUMBER_FORMAT_STYLE_IDS = {
    "0": "3",
    "0.000": "4",
    "0.0000": "5",
}


def natural_sort_key(value: str | Path) -> tuple[Any, ...]:
    """Return one deterministic natural-sort key for strings or paths."""

    text = str(value)
    parts = _NATURAL_RE.split(text.casefold())
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def sort_paths(
    paths: Sequence[Path],
    *,
    sort_by: str = "name",
    root: Path | None = None,
) -> list[Path]:
    """Sort paths naturally by name, relative path, or preserve discovery order."""

    indexed = list(enumerate(paths))
    if sort_by == "none":
        return [path for _index, path in indexed]

    def _key(item: tuple[int, Path]) -> tuple[Any, ...]:
        index, path = item
        rel = None
        if root is not None:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
        rel = rel or path
        if sort_by == "path":
            return natural_sort_key(rel.as_posix()), index
        return (
            natural_sort_key(path.name),
            natural_sort_key(rel.as_posix()),
            index,
        )

    return [path for _index, path in sorted(indexed, key=_key)]


def sort_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_by: str = "name",
) -> list[dict[str, Any]]:
    """Sort summary rows deterministically for CSV and XLSX output."""

    indexed = [(index, dict(row)) for index, row in enumerate(rows)]
    if sort_by == "none":
        return [row for _index, row in indexed]

    def _primary(row: Mapping[str, Any]) -> str:
        return str(row.get("name") or row.get("sample") or "")

    def _path_text(row: Mapping[str, Any]) -> str:
        return str(
            row.get("path")
            or row.get("folder")
            or row.get("json_file")
            or row.get("file")
            or row.get("name")
            or row.get("sample")
            or ""
        )

    def _key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, row = item
        if sort_by == "path":
            return (
                natural_sort_key(_path_text(row)),
                natural_sort_key(str(row.get("file") or "")),
                natural_sort_key(str(row.get("json_file") or "")),
                index,
            )
        return (
            natural_sort_key(_primary(row)),
            natural_sort_key(str(row.get("file") or "")),
            natural_sort_key(str(row.get("json_file") or "")),
            index,
        )

    return [row for _index, row in sorted(indexed, key=_key)]


def ordered_columns(
    rows: Sequence[Mapping[str, Any]],
    preferred_columns: Sequence[str] = (),
) -> list[str]:
    """Return preferred columns first, then deterministic natural extras."""

    seen: set[str] = set()
    ordered: list[str] = []
    for column in preferred_columns:
        if column not in seen:
            ordered.append(column)
            seen.add(column)
    extras = sorted(
        {key for row in rows for key in row.keys() if key not in seen},
        key=natural_sort_key,
    )
    ordered.extend(extras)
    return ordered


def _coerce_table_cell(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv_table(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    preferred_columns: Sequence[str] = (),
    exact_columns: Sequence[str] | None = None,
    *,
    sort_by: str = "name",
) -> Path | None:
    """Write one CSV table with deterministic columns and row ordering."""

    normalized = [
        {key: _coerce_table_cell(value) for key, value in row.items()}
        for row in sort_rows(rows, sort_by=sort_by)
    ]
    if not normalized:
        return None
    fieldnames = list(exact_columns) if exact_columns is not None else ordered_columns(normalized, preferred_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized:
            writer.writerow({column: row.get(column) for column in fieldnames})
    return path


def _column_letter(index: int) -> str:
    value = index
    letters: list[str] = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    base = _INVALID_SHEET_CHARS_RE.sub("_", name).strip() or "Sheet"
    base = base[:31]
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = f"{base[: max(1, 31 - len(tail))]}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


def _cell_reference(row: int, column: int) -> str:
    return f"{_column_letter(column)}{row}"


def _is_wrap_column(column_name: str) -> bool:
    name = column_name.casefold()
    return any(token in name for token in ("path", "error", "folder", "json"))


def xlsx_number_format_for_column(column_name: str) -> str | None:
    """Return one reusable XLSX display format for a semantic column name.

    Raw coverage fractions intentionally do not match the geometric-fraction
    rule.  Only explicitly normalized radius/radial coordinates receive the
    three-decimal presentation format.
    """

    name = column_name.casefold()
    if name in {"count", "index"} or name.endswith(("_count", "_index")):
        return "0"
    if name.endswith("_um"):
        return "0.0000"
    if name.endswith(("_pct", "_pp", "_pp_per_r")):
        return "0.000"
    if "completeness" in name or name.endswith("_deg"):
        return "0.000"
    if name in _XLSX_GEOMETRIC_FRACTION_COLUMNS or name.endswith("_radius_fraction"):
        return "0.000"
    return None


def _column_width(values: Sequence[Any], header: str) -> float:
    lengths = [len(header)]
    for value in values:
        if value is None:
            continue
        lengths.append(len(str(value)))
    width = max(lengths, default=8) + 2
    return float(min(max(width, 8), 60))


def _styles_xml() -> bytes:
    root = Element(
        "styleSheet",
        xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    )
    num_fmts = SubElement(root, "numFmts", count="2")
    SubElement(num_fmts, "numFmt", numFmtId="164", formatCode="0.000")
    SubElement(num_fmts, "numFmt", numFmtId="165", formatCode="0.0000")
    fonts = SubElement(root, "fonts", count="2")
    font = SubElement(fonts, "font")
    SubElement(font, "sz", val="11")
    SubElement(font, "name", val="Calibri")
    header_font = SubElement(fonts, "font")
    SubElement(header_font, "b")
    SubElement(header_font, "sz", val="11")
    SubElement(header_font, "name", val="Calibri")
    fills = SubElement(root, "fills", count="2")
    SubElement(SubElement(fills, "fill"), "patternFill", patternType="none")
    SubElement(SubElement(fills, "fill"), "patternFill", patternType="gray125")
    borders = SubElement(root, "borders", count="1")
    border = SubElement(borders, "border")
    for edge in ("left", "right", "top", "bottom", "diagonal"):
        SubElement(border, edge)
    SubElement(root, "cellStyleXfs", count="1")
    cell_style_xf = SubElement(root.find("cellStyleXfs"), "xf", numFmtId="0", fontId="0", fillId="0", borderId="0")
    cell_xfs = SubElement(root, "cellXfs", count="6")
    SubElement(cell_xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0", xfId="0")
    header_xf = SubElement(cell_xfs, "xf", numFmtId="0", fontId="1", fillId="0", borderId="0", xfId="0", applyFont="1", applyAlignment="1")
    SubElement(header_xf, "alignment", horizontal="center", vertical="center", wrapText="1")
    wrap_xf = SubElement(cell_xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0", xfId="0", applyAlignment="1")
    SubElement(wrap_xf, "alignment", vertical="top", wrapText="1")
    SubElement(cell_xfs, "xf", numFmtId="1", fontId="0", fillId="0", borderId="0", xfId="0", applyNumberFormat="1")
    SubElement(cell_xfs, "xf", numFmtId="164", fontId="0", fillId="0", borderId="0", xfId="0", applyNumberFormat="1")
    SubElement(cell_xfs, "xf", numFmtId="165", fontId="0", fillId="0", borderId="0", xfId="0", applyNumberFormat="1")
    SubElement(root, "cellStyles", count="1")
    SubElement(root.find("cellStyles"), "cellStyle", name="Normal", xfId="0", builtinId="0")
    return tostring(root, encoding="utf-8", xml_declaration=True)


def _worksheet_xml(
    sheet_name: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    semantic_number_formats: bool = False,
) -> bytes:
    sheet = Element(
        "worksheet",
        xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        attrib={
            "xmlns:r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        },
    )
    sheet_views = SubElement(sheet, "sheetViews")
    sheet_view = SubElement(sheet_views, "sheetView", workbookViewId="0")
    SubElement(
        sheet_view,
        "pane",
        ySplit="1",
        topLeftCell="A2",
        activePane="bottomLeft",
        state="frozen",
    )

    if columns:
        cols = SubElement(sheet, "cols")
        for index, column in enumerate(columns, start=1):
            values = [row.get(column) for row in rows]
            SubElement(
                cols,
                "col",
                min=str(index),
                max=str(index),
                width=f"{_column_width(values, column):.2f}",
                customWidth="1",
            )

    sheet_data = SubElement(sheet, "sheetData")
    header_row = SubElement(sheet_data, "row", r="1")
    for column_index, column_name in enumerate(columns or ("Message",), start=1):
        cell = SubElement(
            header_row,
            "c",
            r=_cell_reference(1, column_index),
            t="inlineStr",
            s="1",
        )
        inline = SubElement(cell, "is")
        SubElement(inline, "t").text = column_name

    if not rows and not columns:
        data_row = SubElement(sheet_data, "row", r="2")
        cell = SubElement(data_row, "c", r="A2", t="inlineStr", s="2")
        inline = SubElement(cell, "is")
        SubElement(inline, "t").text = "No rows"
    else:
        for row_index, row in enumerate(rows, start=2):
            row_element = SubElement(sheet_data, "row", r=str(row_index))
            for column_index, column_name in enumerate(columns, start=1):
                value = _coerce_table_cell(row.get(column_name))
                if value is None:
                    continue
                cell_ref = _cell_reference(row_index, column_index)
                style_id = None
                if semantic_number_formats and isinstance(value, (int, float)) and not isinstance(value, bool):
                    number_format = xlsx_number_format_for_column(column_name)
                    if number_format is not None:
                        style_id = _XLSX_NUMBER_FORMAT_STYLE_IDS[number_format]
                elif _is_wrap_column(column_name) or len(str(value)) > 48:
                    style_id = "2"
                if isinstance(value, bool):
                    cell = SubElement(
                        row_element,
                        "c",
                        r=cell_ref,
                        t="b",
                        **({"s": style_id} if style_id is not None else {}),
                    )
                    SubElement(cell, "v").text = "1" if value else "0"
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    cell = SubElement(
                        row_element,
                        "c",
                        r=cell_ref,
                        **({"s": style_id} if style_id is not None else {}),
                    )
                    SubElement(cell, "v").text = str(value)
                else:
                    cell = SubElement(
                        row_element,
                        "c",
                        r=cell_ref,
                        t="inlineStr",
                        **({"s": style_id} if style_id is not None else {}),
                    )
                    inline = SubElement(cell, "is")
                    SubElement(inline, "t").text = str(value)

    last_column = _column_letter(max(1, len(columns)))
    last_row = max(1, len(rows) + 1)
    SubElement(sheet, "autoFilter", ref=f"A1:{last_column}{last_row}")
    return tostring(sheet, encoding="utf-8", xml_declaration=True)


def write_xlsx_workbook(
    sheets: Mapping[str, Sequence[Mapping[str, Any]]],
    path: Path,
    preferred_columns: Mapping[str, Sequence[str]] | None = None,
    exact_columns: Mapping[str, Sequence[str]] | None = None,
    *,
    semantic_number_format_sheets: Sequence[str] = (),
    sort_by: str = "name",
) -> Path | None:
    """Write one XLSX workbook with one or more sheets."""

    preferred_columns = preferred_columns or {}
    exact_columns = exact_columns or {}
    semantic_number_format_sheets = frozenset(semantic_number_format_sheets)
    normalized: list[tuple[str, list[dict[str, Any]], list[str], bool]] = []
    used_names: set[str] = set()
    for requested_name, rows in sheets.items():
        ordered_rows = [
            {key: _coerce_table_cell(value) for key, value in row.items()}
            for row in sort_rows(rows, sort_by=sort_by)
        ]
        columns = (
            list(exact_columns[requested_name])
            if requested_name in exact_columns
            else ordered_columns(ordered_rows, preferred_columns.get(requested_name, ()))
        )
        if not ordered_rows and not columns:
            continue
        sheet_name = _sanitize_sheet_name(requested_name, used_names)
        normalized.append(
            (
                sheet_name,
                ordered_rows,
                columns,
                requested_name in semantic_number_format_sheets,
            )
        )
    if not normalized:
        normalized.append(("Empty", [], [], False))

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Element(
        "workbook",
        xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        attrib={
            "xmlns:r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        },
    )
    sheets_element = SubElement(workbook, "sheets")
    workbook_rels = Element(
        "Relationships",
        xmlns="http://schemas.openxmlformats.org/package/2006/relationships",
    )
    content_types = Element(
        "Types",
        xmlns="http://schemas.openxmlformats.org/package/2006/content-types",
    )
    SubElement(content_types, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    SubElement(content_types, "Default", Extension="xml", ContentType="application/xml")
    SubElement(content_types, "Override", PartName="/xl/workbook.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
    SubElement(content_types, "Override", PartName="/xl/styles.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml")
    for index, (sheet_name, _rows, _columns, _semantic_formats) in enumerate(normalized, start=1):
        SubElement(
            sheets_element,
            "sheet",
            name=sheet_name,
            sheetId=str(index),
            attrib={"r:id": f"rId{index}"},
        )
        SubElement(
            workbook_rels,
            "Relationship",
            Id=f"rId{index}",
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
            Target=f"worksheets/sheet{index}.xml",
        )
        SubElement(
            content_types,
            "Override",
            PartName=f"/xl/worksheets/sheet{index}.xml",
            ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
        )
    SubElement(
        workbook_rels,
        "Relationship",
        Id=f"rId{len(normalized) + 1}",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles",
        Target="styles.xml",
    )

    package_rels = Element(
        "Relationships",
        xmlns="http://schemas.openxmlformats.org/package/2006/relationships",
    )
    SubElement(
        package_rels,
        "Relationship",
        Id="rId1",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
        Target="xl/workbook.xml",
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", tostring(content_types, encoding="utf-8", xml_declaration=True))
        archive.writestr("_rels/.rels", tostring(package_rels, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/workbook.xml", tostring(workbook, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/_rels/workbook.xml.rels", tostring(workbook_rels, encoding="utf-8", xml_declaration=True))
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, (sheet_name, rows, columns, semantic_formats) in enumerate(normalized, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(
                    sheet_name,
                    rows,
                    columns,
                    semantic_number_formats=semantic_formats,
                ),
            )
    return path
