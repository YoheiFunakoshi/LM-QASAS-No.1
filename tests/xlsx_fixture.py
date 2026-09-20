"""Tiny, explicitly synthetic OOXML fixture; never copies a research workbook."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

FIELDS = ('V','V_function','D','D_function','J','J_function','C','C_function','CDR3','frame','count')


def synthetic_row(**changes):
    row = dict(zip(FIELDS, ('IGHV1-2*01','F','IGHD1-1*01','F','IGHJ4*01','F',
                            'IGHG1*01','F','CASSW','in-frame',10)))
    row.update(changes)
    return row


def synthetic_xlsx(rows=None, *, reported_dimension='A1:CP2', overrides=None,
                   sheet_names=('PRINT_hIGH','Back_data'), front_overrides=None):
    rows = [synthetic_row()] if rows is None else rows
    cells = {'F1':'7:All Data', 'S1':'8:In frame Data', 'S2':'Ranking',
             'B4':'Total reads', 'B5':'Assigned reads', 'B6':'In frame', 'B7':'Unique reads (In frame)'}
    for number, row in enumerate(rows, 1):
        for column, key in zip('GHIJKLMNOPQ', FIELDS):
            cells[f'{column}{number}'] = row.get(key)
    numeric=lambda r:r['count'] if type(r.get('count')) in (int,float) else 0
    assigned=sum(numeric(row) for row in rows)
    cells.update(C4=assigned+100, C5=assigned,
                 C6=sum(numeric(r) for r in rows if r['frame']=='in-frame'),
                 C7=len({tuple(r[k] for k in ('V','D','J','CDR3','C')) for r in rows if r['frame']=='in-frame'}))
    cells.update(overrides or {})
    front={'F1':'Repertoire Analysis Report (Human IGH)', 'F5':'SYNTHETIC FIXTURE Sheet ver. hIGH20181210'}
    front.update(front_overrides or {})
    def sheet(values, dimension):
        grouped={}
        for address,value in values.items():
            if value is None: continue
            number=int(''.join(c for c in address if c.isdigit()))
            if isinstance(value,tuple) and value[0]=='formula':
                cell=f'<c r="{address}"><f>{escape(value[1])}</f><v>1</v></c>'
            elif isinstance(value,str):
                cell=f'<c r="{address}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            else:
                cell=f'<c r="{address}"><v>{value}</v></c>'
            grouped.setdefault(number,[]).append(cell)
        content=''.join(f'<row r="{n}">'+''.join(v)+'</row>' for n,v in sorted(grouped.items()))
        return f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="{dimension}"/><sheetData>{content}</sheetData></worksheet>'
    out=BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in (1,2))+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(sheet_names,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in (1,2))+'</Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',sheet(front,'A1:G7'))
        z.writestr('xl/worksheets/sheet2.xml',sheet(cells,reported_dimension))
    return out.getvalue()
