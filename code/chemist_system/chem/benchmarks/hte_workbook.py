"""Small read-only OOXML reader for the two frozen HTE input workbooks.

No formula evaluation, Excel automation, type coercion of nominal strings, or
environment mutation. Numeric cached cell values are kept, blanks stay None.
"""
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}


def column_index(label):
    result = 0
    for char in label:
        if not 'A' <= char <= 'Z':
            raise ValueError('invalid column')
        result = result * 26 + ord(char) - ord('A') + 1
    return result - 1


def sheets(path, selected=None):
    with ZipFile(Path(path)) as book:
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            root = ET.fromstring(book.read('xl/sharedStrings.xml'))
            strings = [''.join(el.itertext()) for el in root.findall('s:si', NS)]
        relationships = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships}
        workbook = ET.fromstring(book.read('xl/workbook.xml'))
        for sheet in workbook.findall('s:sheets/s:sheet', NS):
            if selected is not None and sheet.attrib['name'] != selected:
                continue
            target = targets[sheet.attrib['{' + NS['r'] + '}id']]
            name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/' + target)
            root = ET.fromstring(book.read(name))
            rows = []
            for row in root.findall('s:sheetData/s:row', NS):
                values = {}
                for cell in row.findall('s:c', NS):
                    match = re.fullmatch(r'([A-Z]+)(\d+)', cell.attrib['r'])
                    index = column_index(match.group(1))
                    kind = cell.attrib.get('t')
                    value = cell.find('s:v', NS)
                    if kind == 'inlineStr':
                        inline = cell.find('s:is', NS)
                        parsed = ''.join(inline.itertext()) if inline is not None else ''
                    elif value is None or value.text is None:
                        parsed = None
                    elif kind == 's':
                        parsed = strings[int(value.text)]
                    elif kind in ('str', 'e'):
                        parsed = value.text
                    elif kind == 'b':
                        parsed = bool(int(value.text))
                    else:
                        parsed = float(value.text)
                        if parsed.is_integer():
                            parsed = int(parsed)
                    values[index] = parsed
                if values:
                    rows.append({'excel_row': int(row.attrib['r']),
                                 'values': [values.get(i) for i in range(max(values)+1)]})
            yield sheet.attrib['name'], rows


def table(path, sheet_name):
    rows = next(rows for name, rows in sheets(path, selected=sheet_name) if name == sheet_name)
    if not rows:
        raise ValueError('empty sheet')
    headers = rows[0]['values']
    if any(not isinstance(k, str) or not k for k in headers) or len(set(headers)) != len(headers):
        raise ValueError('expected unique named header cells')
    output = []
    for row in rows[1:]:
        values = row['values']
        if len(values) > len(headers):
            if any(value not in (None, '') for value in values[len(headers):]):
                raise ValueError('nonempty cell beyond header')
            values = values[:len(headers)]
        values = values + [None] * (len(headers)-len(values))
        output.append({'source_excel_row': row['excel_row'], **dict(zip(headers, values))})
    return output
