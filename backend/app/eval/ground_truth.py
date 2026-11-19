"""Create the ground-truth workbook template to be filled BY HAND for the real test room.

    python -m app.eval.ground_truth data/eval/ground_truth.xlsx

Nothing in the template is pre-filled with data: it contains headers, instructions and empty rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

SHEETS = {
    "README": [["How to fill this sheet (all values MEASURED/READ BY A HUMAN in the real test room; never copy system output)"],
               ["Books: one row per physical book on the shelves. legible=Y if a human can read the title from the spine in a normal photo."],
               ["Spines_measured: 20 books measured with a ruler (cm). title must match a Books row."],
               ["Prices_checked: 15 books whose price you checked by hand today on a listing; record URL + date + currency + condition."],
               ["NonBook: every non-book object in the room with its category (bookshelf, cabinet, desk, chair, lamp, rug, artwork, portrait, electronics, coffee_machine, decorative, table, sofa, mirror, appliance)."],
               ["Room: tape-measured dimensions. Wall area = perimeter x height (gross, no openings subtracted) unless you note otherwise."]],
    "Books": [["id", "shelf_unit", "shelf_row", "position_left_to_right", "legible (Y/N)", "title (as printed)", "author", "notes"]],
    "Spines_measured": [["title", "height_cm", "thickness_cm", "measured_with"]],
    "Prices_checked": [["title", "isbn", "kind (replacement|used)", "price", "currency", "condition", "source_url", "checked_on (YYYY-MM-DD)"]],
    "NonBook": [["id", "category", "description", "notes"]],
    "Room": [["field", "value", "unit", "how_measured"],
             ["length", "", "m", ""], ["width", "", "m", ""], ["ceiling_height", "", "m", ""], ["floor_area", "", "m2", ""],
             ["wall_area", "", "m2", ""], ["shelved_wall_area", "", "m2", ""], ["true_book_count", "", "books", "counted by two people"]],
}


def build(path: Path) -> None:
    wb = Workbook(); wb.remove(wb.active)
    for name, rows in SHEETS.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
        if name != "README":
            for c in ws[1]:
                c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
        for i in range(1, 9):
            ws.column_dimensions[get_column_letter(i)].width = 26
    path.parent.mkdir(parents=True, exist_ok=True); wb.save(path)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/ground_truth.xlsx")
    build(out); print("wrote", out)
