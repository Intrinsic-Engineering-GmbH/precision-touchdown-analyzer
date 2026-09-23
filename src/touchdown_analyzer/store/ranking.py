"""The ranking as the board shows it, written out as Excel and PDF. Stdlib only.

Whatever changes what the board shows - a landing confirmed or reopened, a
pilot named, the rules edited, a new landing from the analyser - refreshes
``ranking.xlsx`` and ``ranking.pdf`` next to the session's ``landings.json``
(:meth:`ReviewService.export`), so the sheet the organiser hands out is never
behind the screen. The grouping and order here mirror ``board.html``: a
pilot's confirmed landings added up, best total first, then the landings
still waiting for the judge.

Both files are produced without libraries: a .xlsx is a zip of XML parts,
and a table of text in Helvetica is a few hundred lines of PDF. That keeps
the recorder's stdlib-only footprint and the installer free of yet another
dependency.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from touchdown_analyzer.store.scoring import ScoringRules

XLSX_NAME = "ranking.xlsx"
PDF_NAME = "ranking.pdf"


@dataclass(slots=True)
class Group:
    """One row of the ranking: a pilot (or, unnamed, an aircraft)."""

    key: str
    name: str
    pilot: str
    aircraft: list[str] = field(default_factory=list)
    landings: list[dict[str, Any]] = field(default_factory=list)
    total: float = 0.0


@dataclass(slots=True)
class Ranking:
    ranked: list[Group]
    pending: list[dict[str, Any]]

    @property
    def confirmed(self) -> int:
        return sum(len(g.landings) for g in self.ranked)


def _when(landing: dict[str, Any]) -> str:
    return landing.get("touchdown_utc") or landing.get("first_utc") or ""


def rank(landings: list[dict[str, Any]]) -> Ranking:
    """Group and order scored landings the way the board does.

    Rejected entries and anything that is not a landing are left out. The
    pilot is what the judge typed; a landing without one falls back to the
    aircraft, and one without either stands alone.
    """
    shown = [x for x in landings if x.get("kind") == "landing" and x.get("status") != "rejected"]
    groups: dict[str, Group] = {}
    for x in sorted((x for x in shown if x.get("status") == "confirmed"), key=_when):
        pilot = (x.get("pilot") or "").strip()
        registration = x.get("registration") or ""
        if pilot:
            key = "p:" + pilot.lower()
        elif registration:
            key = "a:" + registration
        else:
            key = "#" + str(x.get("id"))
        group = groups.get(key)
        if group is None:
            group = groups[key] = Group(key, pilot or registration or "unknown", pilot)
        group.landings.append(x)
        group.total += x.get("score") or 0
        if pilot:
            group.name = pilot  # the spelling of the latest entry
        craft = " ".join(s for s in (registration, x.get("competition_number")) if s)
        if craft and craft not in group.aircraft:
            group.aircraft.append(craft)
        if not pilot and not group.aircraft and x.get("aircraft_type"):
            group.aircraft.append(x["aircraft_type"])
    ranked = sorted(groups.values(), key=lambda g: (-g.total, -len(g.landings), g.name.casefold()))
    pending = sorted((x for x in shown if x.get("status") != "confirmed"), key=_when)
    return Ranking(ranked, pending)


# -- what goes into the cells -------------------------------------------------


def local_time(utc: str | None, fmt: str = "%H:%M:%S") -> str:
    if not utc:
        return ""
    try:
        return datetime.fromisoformat(utc).astimezone().strftime(fmt)
    except ValueError:
        return utc


def session_title(session: str) -> str:
    try:
        return datetime.fromisoformat(session[:10]).strftime("%A, %d %B %Y")
    except ValueError:
        return session


def offset_text(landing: dict[str, Any]) -> str:
    """``-1.9 m`` for a measured landing, otherwise the label (``< -12 m``)."""
    metres = landing.get("scored_longitudinal_m")
    if landing.get("outcome") != "measured" or metres is None:
        return str(landing.get("label") or "")
    return f"{metres:+.1f} m"


def points_text(score: float | None, rules: ScoringRules) -> str:
    return "-" if score is None else f"{score:.{rules.decimals}f}"


def wait_text(landing: dict[str, Any]) -> str:
    return "judge picks frame" if landing.get("outcome") == "unseen" else "being verified"


def rules_text(rules: ScoringRules) -> str:
    line = f"{rules.max_points:g} pts on the line"
    if rules.target_width_m > 0:
        line += f" ({rules.target_width_m:g} m wide, +/-{rules.target_half_width_m:g} m)"
    return (
        f"{rules.name or 'scoring'}: {line}, "
        f"-{rules.short_per_m:g}/m short, -{rules.long_per_m:g}/m long, "
        f"floor {rules.min_points:g}, outside the window {rules.out_of_range_points:g}"
    )


# -- Excel --------------------------------------------------------------------

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{sheets}</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{sheets}</sheets></workbook>"""

_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdS" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
{sheets}</Relationships>"""

# cell styles (cellXfs index): 0 plain, 1 bold, 2 title, 3 number 0 dp, 4 number 1 dp
_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.0"/></numFmts>
<fonts count="3"><font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="14"/><name val="Calibri"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="5">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""

PLAIN, BOLD, TITLE, INT, ONE_DP = 0, 1, 2, 3, 4


def _column(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell(ref: str, value: Any, style: int) -> str:
    if value is None or value == "":
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, int | float):
        return f'<c r="{ref}" s="{style}"><v>{value!r}</v></c>'
    text = escape(str(value)).replace("\n", "&#10;")
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet_xml(rows: list[list[tuple[Any, int]]], widths: list[float]) -> str:
    """A worksheet from rows of ``(value, style)`` cells."""
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
        for i, w in enumerate(widths)
    )
    body = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            _cell(f"{_column(c)}{r}", value, style) for c, (value, style) in enumerate(row)
        )
        body.append(f'<row r="{r}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{cols}</cols><sheetData>{''.join(body)}</sheetData></worksheet>"
    )


def _number_style(rules: ScoringRules) -> int:
    return INT if rules.decimals == 0 else ONE_DP


def _heading_rows(session: str, rules: ScoringRules, ranking: Ranking) -> list[list[tuple]]:
    return [
        [("Live Ranking", TITLE)],
        [(session_title(session), BOLD)],
        [(rules_text(rules), PLAIN)],
        [
            (
                f"{ranking.confirmed} confirmed landings, {len(ranking.ranked)} pilots, "
                f"{len(ranking.pending)} awaiting the judge - written "
                f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
                PLAIN,
            )
        ],
        [],
    ]


def _ranking_sheet(ranking: Ranking, session: str, rules: ScoringRules) -> str:
    n = _number_style(rules)
    rows = _heading_rows(session, rules, ranking)
    rows.append([(h, BOLD) for h in ("Rank", "Pilot", "Aircraft", "Landings", "Total")])
    for i, g in enumerate(ranking.ranked, start=1):
        rows.append(
            [
                (i, INT),
                (g.name, PLAIN),
                (" / ".join(g.aircraft), PLAIN),
                (len(g.landings), INT),
                (round(g.total, rules.decimals), n),
            ]
        )
    if ranking.pending:
        rows += [[], [("Awaiting the judge", BOLD)]]
        rows.append([(h, BOLD) for h in ("", "Pilot / aircraft", "Aircraft", "Time", "State")])
        for x in ranking.pending:
            rows.append(
                [
                    ("", PLAIN),
                    (x.get("pilot") or x.get("registration") or "unknown", PLAIN),
                    (
                        " ".join(
                            s
                            for s in (
                                x.get("registration") if x.get("pilot") else "",
                                x.get("competition_number"),
                                x.get("aircraft_type"),
                            )
                            if s
                        ),
                        PLAIN,
                    ),
                    (local_time(_when(x)), PLAIN),
                    (wait_text(x), PLAIN),
                ]
            )
    return _sheet_xml(rows, [7, 28, 24, 10, 10])


def _landings_sheet(ranking: Ranking, session: str, rules: ScoringRules) -> str:
    """Every confirmed landing on its own row, in ranking order."""
    n = _number_style(rules)
    rows = _heading_rows(session, rules, ranking)
    rows.append(
        [
            (h, BOLD)
            for h in (
                "Rank",
                "Pilot",
                "Registration",
                "Comp. no.",
                "Type",
                "Time",
                "Offset m",
                "Result",
                "Points",
                "Landing",
                "Note",
            )
        ]
    )
    for i, g in enumerate(ranking.ranked, start=1):
        for x in g.landings:
            metres = x.get("scored_longitudinal_m") if x.get("outcome") == "measured" else None
            rows.append(
                [
                    (i, INT),
                    (g.name, PLAIN),
                    (x.get("registration") or "", PLAIN),
                    (x.get("competition_number") or "", PLAIN),
                    (x.get("aircraft_type") or "", PLAIN),
                    (local_time(_when(x)), PLAIN),
                    (None if metres is None else round(metres, 2), ONE_DP),
                    (offset_text(x), PLAIN),
                    (x.get("score"), n),
                    (x.get("id") or "", PLAIN),
                    (x.get("note") or "", PLAIN),
                ]
            )
    return _sheet_xml(rows, [7, 28, 14, 10, 14, 10, 10, 12, 8, 9, 40])


def write_xlsx(path: Path, ranking: Ranking, session: str, rules: ScoringRules) -> None:
    sheets = {
        "Ranking": _ranking_sheet(ranking, session, rules),
        "Landings": _landings_sheet(ranking, session, rules),
    }
    tmp = path.with_suffix(".xlsx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            _CONTENT_TYPES.format(
                sheets="".join(
                    f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    for i in range(1, len(sheets) + 1)
                )
            ),
        )
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr(
            "xl/workbook.xml",
            _WORKBOOK.format(
                sheets="".join(
                    f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                    for i, name in enumerate(sheets, start=1)
                )
            ),
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            _WORKBOOK_RELS.format(
                sheets="".join(
                    f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                    for i in range(1, len(sheets) + 1)
                )
            ),
        )
        zf.writestr("xl/styles.xml", _STYLES)
        for i, xml in enumerate(sheets.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", xml)
    os.replace(tmp, path)


# -- PDF ----------------------------------------------------------------------

PAGE_W, PAGE_H = 595.0, 842.0  # A4 portrait, points
MARGIN = 40.0
LINE = 14.0
PAD = 5.0  # inside a table cell, either side

# Helvetica advance widths are not to hand without the AFM; this average is
# close enough to keep columns from overlapping.
_CHAR_W = 0.52


def _text_width(text: str, size: float) -> float:
    return len(text) * size * _CHAR_W


def _pdf_string(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _fit(text: str, width: float, size: float) -> str:
    if _text_width(text, size) <= width:
        return text
    keep = max(1, int(width / (size * _CHAR_W)) - 1)
    return text[:keep] + "…"


@dataclass(slots=True)
class _Column:
    title: str
    width: float
    align: str = "left"  # left | right


class _Pages:
    """Content streams of a document, one per page, with a running cursor."""

    def __init__(self, title: str, subtitle: list[str]) -> None:
        self.title = title
        self.subtitle = subtitle
        self.pages: list[list[bytes]] = []
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        self.pages.append([])
        self.y = PAGE_H - MARGIN
        self.text(MARGIN, self.y - 14, self.title, 16, bold=True)
        self.y -= 34
        for line in self.subtitle:
            self.text(MARGIN, self.y, line, 9)
            self.y -= 12
        self.y -= 8

    def text(self, x: float, y: float, s: str, size: float, *, bold: bool = False) -> None:
        font = b"/F2" if bold else b"/F1"
        self.pages[-1].append(
            b"BT " + font + b" %.1f Tf %.1f %.1f Td " % (size, x, y) + _pdf_string(s) + b" Tj ET\n"
        )

    def rule(self, y: float) -> None:
        self.pages[-1].append(
            b"0.75 w %.1f %.1f m %.1f %.1f l S\n" % (MARGIN, y, PAGE_W - MARGIN, y)
        )

    def need(self, height: float) -> None:
        if self.y - height < MARGIN:
            self._new_page()

    def heading(self, s: str) -> None:
        self.need(3 * LINE)
        self.y -= 6
        self.text(MARGIN, self.y, s, 12, bold=True)
        self.y -= LINE + 2

    def table(self, columns: list[_Column], rows: list[list[str]], *, size: float = 9.5) -> None:
        def header() -> None:
            self.need(2 * LINE)
            self._row(columns, [c.title for c in columns], size, bold=True)
            self.rule(self.y + LINE - 4)

        header()
        for row in rows:
            if self.y - LINE < MARGIN:
                self._new_page()
                header()
            self._row(columns, row, size)
        self.y -= 4

    def _row(
        self, columns: list[_Column], cells: list[str], size: float, *, bold: bool = False
    ) -> None:
        x = MARGIN
        for column, cell in zip(columns, cells, strict=False):
            shown = _fit(cell, column.width - 2 * PAD, size)
            if column.align == "right":
                self.text(
                    x + column.width - PAD - _text_width(shown, size), self.y, shown, size, bold=bold
                )
            else:
                self.text(x + PAD, self.y, shown, size, bold=bold)
            x += column.width
        self.y -= LINE

    def paragraph(self, s: str, size: float = 9.5) -> None:
        self.need(LINE)
        self.text(MARGIN, self.y, s, size)
        self.y -= LINE


def _pdf_bytes(pages: _Pages) -> bytes:
    """Assemble the objects and the xref table."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font1 = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    font2 = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    )
    pages_id = len(objects) + 1 + 2 * len(pages.pages)  # after every page + content pair
    page_ids = []
    total = len(pages.pages)
    for index, stream in enumerate(pages.pages, start=1):
        footer = b"BT /F1 8 Tf %.1f %.1f Td " % (PAGE_W - MARGIN - 40, MARGIN - 16)
        footer += _pdf_string(f"page {index} of {total}") + b" Tj ET\n"
        content = b"".join(stream) + footer
        content_id = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream")
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.0f %.0f] "
                b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_id, PAGE_W, PAGE_H, font1, font2, content_id)
            )
        )
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    assert add(b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % total) == pages_id
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog,
        xref,
    )
    return bytes(out)


def write_pdf(path: Path, ranking: Ranking, session: str, rules: ScoringRules) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    doc = _Pages(
        f"Live Ranking - {session_title(session)}",
        [
            rules_text(rules),
            f"{ranking.confirmed} confirmed landings, {len(ranking.ranked)} pilots, "
            f"{len(ranking.pending)} awaiting the judge - written {stamp}",
        ],
    )

    doc.heading("Ranking")
    if not ranking.ranked:
        doc.paragraph("No confirmed landings yet.")
    else:
        columns = [
            _Column("#", 24, "right"),
            _Column("Pilot", 135),
            _Column("Aircraft", 110),
            _Column("Landings", 50, "right"),
            _Column("Each landing", 150),
            _Column("Total", 46, "right"),
        ]
        rows = []
        for i, g in enumerate(ranking.ranked, start=1):
            each = " ".join(points_text(x.get("score"), rules) for x in g.landings)
            rows.append(
                [
                    str(i),
                    g.name,
                    " / ".join(g.aircraft),
                    str(len(g.landings)),
                    each,
                    points_text(round(g.total, rules.decimals), rules),
                ]
            )
        doc.table(columns, rows)

        doc.heading("Landings")
        columns = [
            _Column("#", 24, "right"),
            _Column("Pilot", 135),
            _Column("Aircraft", 110),
            _Column("Time", 60),
            _Column("Offset", 60, "right"),
            _Column("Points", 50, "right"),
            _Column("Landing", 76),
        ]
        rows = []
        for i, g in enumerate(ranking.ranked, start=1):
            for x in g.landings:
                craft = " ".join(
                    s for s in (x.get("registration"), x.get("competition_number")) if s
                )
                rows.append(
                    [
                        str(i),
                        g.name,
                        craft,
                        local_time(_when(x)),
                        offset_text(x),
                        points_text(x.get("score"), rules),
                        str(x.get("id") or ""),
                    ]
                )
        doc.table(columns, rows)

    if ranking.pending:
        doc.heading("Awaiting the judge")
        columns = [
            _Column("Pilot / aircraft", 200),
            _Column("Aircraft", 130),
            _Column("Time", 70),
            _Column("State", 115),
        ]
        rows = []
        for x in ranking.pending:
            craft = " ".join(
                s
                for s in (
                    x.get("registration") if x.get("pilot") else "",
                    x.get("competition_number"),
                    x.get("aircraft_type"),
                )
                if s
            )
            rows.append(
                [
                    x.get("pilot") or x.get("registration") or "unknown",
                    craft,
                    local_time(_when(x)),
                    wait_text(x),
                ]
            )
        doc.table(columns, rows)

    doc.y -= 6
    doc.paragraph("Total = the points of all confirmed landings of the pilot added up.", 8)
    doc.paragraph("Offset from the target line: - short, + long.", 8)

    tmp = path.with_suffix(".pdf.tmp")
    tmp.write_bytes(_pdf_bytes(doc))
    os.replace(tmp, path)


# -- both ---------------------------------------------------------------------


def export(
    directory: Path, session: str, landings: list[dict[str, Any]], rules: ScoringRules
) -> tuple[Path, Path]:
    """Write ``ranking.xlsx`` and ``ranking.pdf`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    ranking = rank(landings)
    xlsx, pdf = directory / XLSX_NAME, directory / PDF_NAME
    write_xlsx(xlsx, ranking, session, rules)
    write_pdf(pdf, ranking, session, rules)
    return xlsx, pdf
