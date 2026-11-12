"""Render the same Word paragraphs to print HTML for a Korean-font PDF."""
from html import escape
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('C:/Users/jes/Desktop/MX3_Match_Sim/Manual')
CHECK = ROOT / 'manuals/render_check_20261109'
NAME = 'MX3_HSL_로봇운영매뉴얼_20261109'


def run_html(element):
    pieces = []
    for part in element.iter():
        if part.tag == qn('w:t'):
            pieces.append(escape(part.text or ''))
        elif part.tag == qn('w:br') and part.get(qn('w:type')) != 'page':
            pieces.append('<br>')
    value = ''.join(pieces)
    bold = element.find('.//' + qn('w:b'))
    if bold is not None and bold.get(qn('w:val'), '1') not in ('0', 'false', 'off'):
        value = '<strong>' + value + '</strong>'
    fonts = element.find('.//' + qn('w:rFonts'))
    if fonts is not None and fonts.get(qn('w:ascii')) == 'Consolas':
        value = '<code>' + value + '</code>'
    return value


def main():
    document = Document(OUTPUT / (NAME + '.docx'))
    body = []
    for paragraph in document.paragraphs:
        style = paragraph.style.name.replace(' ', '')
        if any(b.get(qn('w:type')) == 'page' for b in paragraph._p.iter(qn('w:br'))):
            body.append('<div class="pagebreak"></div>')
        parts = []
        for element in paragraph._p:
            if element.tag == qn('w:r'):
                parts.append(run_html(element))
            elif element.tag == qn('w:hyperlink'):
                target = element.get(qn('w:anchor'))
                rid = element.get(qn('r:id'))
                target = '#' + target if target else str(document.part.rels[rid].target_ref)
                parts.append('<a href="%s">%s</a>' % (escape(target, quote=True), run_html(element)))
        tag = 'h' + style[-1] if style.startswith('Heading') else 'p'
        extra = ''
        bookmarks = list(paragraph._p.iter(qn('w:bookmarkStart')))
        if bookmarks:
            extra = ' id="' + escape(bookmarks[0].get(qn('w:name')), quote=True) + '"'
        if paragraph.text == '로봇 운영\n매뉴얼':
            style += ' coverhero'
        elif paragraph.text == '목차':
            style += ' toctitle'
        if any(run.font.size and run.font.size.pt >= 30 for run in paragraph.runs):
            style += ' coverhero'
        body.append('<%s class="%s"%s>%s</%s>' % (tag, style, extra, ''.join(parts) or '&nbsp;', tag))
    css = '''
@page { size: A4; margin: 18mm 20mm 18mm; @bottom-right { content: "MX3 + HSL  /  " counter(page); font: 8pt Arial; color: #637585; } }
* { box-sizing: border-box; }
body { margin: 0; font: 10pt/1.42 "Malgun Gothic", "맑은 고딕", sans-serif; color: #172b3b; }
p { margin: 0 0 6pt; orphans: 2; widows: 2; overflow-wrap: anywhere; }
h1,h2,h3 { color: #173d59; line-height: 1.3; break-after: avoid; overflow-wrap: anywhere; }
h1 { font-size: 18pt; margin: 18pt 0 10pt; }
h1#chapter_0, h1#chapter_1, h1#chapter_9, h1#chapter_10 { break-before: page; margin-top: 0; }
h2 { font-size: 13pt; margin: 14pt 0 8pt; }
h3 { font-size: 11pt; margin: 12pt 0 7pt; }
.coverhero { font-size: 34pt; line-height: 1.3; font-weight: 700; color: #173d59; margin: 30pt 0 24pt; }
.toctitle { font-size: 20pt; font-weight: 700; margin: 0 0 16pt; }
.pagebreak { break-before: page; }
.RecordTitle { font-weight: 700; color: #173d59; margin-top: 9pt; margin-bottom: 3pt; break-after: avoid; }
.Record { margin-bottom: 7pt; }
.RecordDetail { margin-left: 10pt; margin-bottom: 4pt; }
.ManualList { margin-left: 7pt; }
.Command { background: #f0f4f7; color: #16324b; padding: 1pt 7pt; margin: 0; font: 8.3pt/1.3 Consolas,"Malgun Gothic",monospace; white-space: pre-wrap; overflow-wrap: anywhere; break-after: avoid; }
.CommandSpacer { height: 4pt; margin: 0; line-height: 1; font-size: 2pt; }
code { font: 9pt Consolas,"Malgun Gothic",monospace; }
.Command code { font-size: inherit; }
a { color: #14678a; text-decoration: underline; }
'''
    CHECK.mkdir(parents=True, exist_ok=True)
    path = CHECK / 'manual_print.html'
    path.write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>MX3 + HSL 로봇 운영 매뉴얼</title><style>'
                    + css + '</style></head><body>' + '\n'.join(body) + '</body></html>', encoding='utf-8')
    print(path.as_uri())


if __name__ == '__main__':
    main()
