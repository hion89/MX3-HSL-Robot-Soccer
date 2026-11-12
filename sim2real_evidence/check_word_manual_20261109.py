"""Inspect the rendered PDF and verify that commands survived Word conversion."""
import json
from pathlib import Path
import re

from docx import Document
from PIL import Image, ImageDraw
import pymupdf


ROOT = Path(__file__).resolve().parent
OUTPUT = Path('C:/Users/jes/Desktop/MX3_Match_Sim/Manual')
NAME = 'MX3_HSL_로봇운영매뉴얼_20261109'
CHECK = ROOT / 'manuals/render_check_20261109'


def main():
    CHECK.mkdir(parents=True, exist_ok=True)
    document = Document(OUTPUT / (NAME + '.docx'))
    paragraphs = [p.text for p in document.paragraphs]
    for source in ('ROBOT_OPERATION_MANUAL_KO_20261109.md', 'BHUMAN_PORT_AUDIT_KO_20261109.md'):
        text = (ROOT / 'manuals' / source).read_text(encoding='utf-8')
        for block in re.findall(r'```bash\n(.*?)```', text, re.S):
            for line in block.rstrip('\n').split('\n'):
                assert line in paragraphs or not line.strip(), 'Command lost: ' + line
    pdf = pymupdf.open(OUTPUT / (NAME + '.pdf'))
    text = '\n'.join(page.get_text() for page in pdf)
    normalized = re.sub(r'\s+', '', text)
    for required in ('로봇운영', '4대4', 'B-Human', 'goalkeeper', 'field_profile', 'robo08', '목차'):
        assert re.sub(r'\s+', '', required) in normalized, required
    assert '\ufffd' not in text
    near_edges, short_pages = [], []
    thumbnails = []
    for index, page in enumerate(pdf):
        if len(page.get_text().strip()) < 30:
            short_pages.append(index + 1)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 15 or x1 > page.rect.width - 15 or y0 < 4 or y1 > page.rect.height - 4:
                        near_edges.append({'page': index + 1, 'text': span['text'], 'bbox': span['bbox']})
        pix = page.get_pixmap(matrix=pymupdf.Matrix(.42, .42), alpha=False)
        thumb = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        tile = Image.new('RGB', (thumb.width + 12, thumb.height + 30), 'white')
        tile.paste(thumb, (6, 24))
        ImageDraw.Draw(tile).text((8, 5), 'Page %d' % (index + 1), fill='black')
        thumbnails.append(tile)
    width, height = thumbnails[0].size
    columns = 5
    sheet = Image.new('RGB', (width * columns, height * ((len(thumbnails) + columns - 1) // columns)), '#d8dce0')
    for index, tile in enumerate(thumbnails):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    sheet.save(CHECK / 'contact_sheet.png')
    (CHECK / 'pdf_text.txt').write_text(text, encoding='utf-8')
    report = {'pages': len(pdf), 'blank_or_near_blank_pages': short_pages, 'text_near_page_edges': near_edges,
              'word_tables': len(document.tables), 'source_commands_preserved': True,
              'pdf_has_korean_and_required_sections': True,
              'files': [{'name': path.name, 'bytes': path.stat().st_size} for path in OUTPUT.iterdir() if path.is_file()]}
    (CHECK / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print('CONTACT_SHEET:', CHECK / 'contact_sheet.png')


if __name__ == '__main__':
    main()
