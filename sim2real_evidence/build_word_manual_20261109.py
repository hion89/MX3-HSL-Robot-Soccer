"""Create a Korean Word manual; Word itself exports the matching PDF."""
import csv
from pathlib import Path
import re

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'manuals'
OUTPUT = Path('C:/Users/jes/Desktop/MX3_Match_Sim/Manual')
NAME = 'MX3_HSL_로봇운영매뉴얼_20261109'
BLUE = '173D59'
MD = MarkdownIt('commonmark').enable('table')


def font(run, name='맑은 고딕', size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '맑은 고딕')
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def field(paragraph, code, placeholder=''):
    run = paragraph.add_run()
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instruction = OxmlElement('w:instrText')
    instruction.set(qn('xml:space'), 'preserve')
    instruction.text = ' ' + code + ' '
    separate = OxmlElement('w:fldChar')
    separate.set(qn('w:fldCharType'), 'separate')
    text = OxmlElement('w:t')
    text.text = placeholder
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    for element in (begin, instruction, separate, text, end):
        run._r.append(element)


def hyperlink(paragraph, text, target):
    if not target.startswith(('https://', 'http://')):
        font(paragraph.add_run(text))
        return
    rid = paragraph.part.relate_to(target,
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), rid)
    run = OxmlElement('w:r')
    props = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '14678A')
    props.append(color)
    underline = OxmlElement('w:u')
    underline.set(qn('w:val'), 'single')
    props.append(underline)
    fonts = OxmlElement('w:rFonts')
    fonts.set(qn('w:eastAsia'), '맑은 고딕')
    props.append(fonts)
    run.append(props)
    value = OxmlElement('w:t')
    value.text = text
    run.append(value)
    link.append(run)
    paragraph._p.append(link)


def inline(paragraph, tokens):
    bold = italic = False
    i = 0
    while i < len(tokens or []):
        token = tokens[i]
        if token.type == 'strong_open':
            bold = True
        elif token.type == 'strong_close':
            bold = False
        elif token.type == 'em_open':
            italic = True
        elif token.type == 'em_close':
            italic = False
        elif token.type == 'link_open':
            label, target = [], token.attrGet('href') or ''
            i += 1
            while i < len(tokens) and tokens[i].type != 'link_close':
                if tokens[i].type in ('text', 'code_inline'):
                    label.append(tokens[i].content)
                i += 1
            hyperlink(paragraph, ''.join(label), target)
        elif token.type in ('softbreak', 'hardbreak'):
            paragraph.add_run(' ' if token.type == 'softbreak' else '\n')
        elif token.type in ('text', 'code_inline'):
            run = paragraph.add_run(token.content)
            font(run, name='Consolas' if token.type == 'code_inline' else '맑은 고딕',
                 size=9 if token.type == 'code_inline' else None, bold=bold)
            run.italic = italic
        i += 1


def add_inline(doc, content, style=None):
    paragraph = doc.add_paragraph(style=style)
    inline(paragraph, MD.parseInline(content)[0].children)
    return paragraph


def codeblock(doc, text, language='bash'):
    # Preserve the commands exactly. Word wraps long lines visually without modifying text.
    lines = text.rstrip('\n').split('\n')
    for index, line in enumerate(lines):
        paragraph = doc.add_paragraph(style='Command')
        paragraph.paragraph_format.keep_with_next = index < len(lines) - 1
        font(paragraph.add_run(line or ' '), name='Consolas', size=8.3, color='16324B')
    doc.add_paragraph(style='CommandSpacer')


def row_records(doc, rows):
    """Turn Markdown tables into full-width records, without narrow Word columns."""
    if len(rows) < 2:
        return
    headers = rows[0]
    for row in rows[1:]:
        if len(row) == 2:
            paragraph = doc.add_paragraph(style='Record')
            inline(paragraph, MD.parseInline(row[0])[0].children)
            for run in paragraph.runs:
                font(run, bold=True, color=BLUE)
            font(paragraph.add_run('：'), bold=True, color=BLUE)
            inline(paragraph, MD.parseInline(row[1])[0].children)
        else:
            paragraph = doc.add_paragraph(style='RecordTitle')
            inline(paragraph, MD.parseInline(row[0])[0].children)
            for index, cell in enumerate(row[1:], 1):
                paragraph = doc.add_paragraph(style='RecordDetail')
                font(paragraph.add_run(headers[index] + '：'), bold=True)
                inline(paragraph, MD.parseInline(cell)[0].children)


def markdown(doc, text, mode='main'):
    tokens = MD.parse(text)
    i, list_depth, ordered, item_counters = 0, 0, [], []
    while i < len(tokens):
        token = tokens[i]
        if token.type == 'heading_open':
            original = int(token.tag[1:])
            content = tokens[i + 1].content
            if original != 1:
                level = original - 1 if mode == 'main' else original
                if mode == 'audit' and original == 2:
                    content = '9.' + content
                paragraph = doc.add_heading(level=min(level, 3))
                inline(paragraph, MD.parseInline(content)[0].children)
                for run in paragraph.runs:
                    run.bold = True
                if mode == 'main' and original == 2 and content.startswith('1.'):
                    paragraph.paragraph_format.page_break_before = True
            i += 3
            continue
        if token.type in ('fence', 'code_block'):
            codeblock(doc, token.content, token.info)
        elif token.type == 'table_open':
            rows, row = [], None
            i += 1
            while i < len(tokens) and tokens[i].type != 'table_close':
                current = tokens[i]
                if current.type == 'tr_open':
                    row = []
                elif current.type == 'inline' and row is not None:
                    row.append(current.content)
                elif current.type == 'tr_close':
                    rows.append(row)
                    row = None
                i += 1
            row_records(doc, rows)
        elif token.type in ('bullet_list_open', 'ordered_list_open'):
            list_depth += 1
            ordered.append(token.type == 'ordered_list_open')
            item_counters.append((token.attrGet('start') or 1) - 1)
        elif token.type in ('bullet_list_close', 'ordered_list_close'):
            list_depth -= 1
            ordered.pop()
            item_counters.pop()
        elif token.type == 'list_item_open':
            item_counters[-1] += 1
        elif token.type == 'paragraph_open':
            following = tokens[i + 1]
            if following.type == 'inline':
                paragraph = doc.add_paragraph(style='ManualList' if list_depth else 'Normal')
                if list_depth:
                    prefix = str(item_counters[-1]) + '. ' if ordered[-1] else '• '
                    font(paragraph.add_run(prefix), bold=ordered[-1])
                inline(paragraph, following.children)
                i += 2
        i += 1


def prepare_styles(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
    sec.left_margin, sec.right_margin = Cm(2), Cm(2)
    sec.header_distance, sec.footer_distance = Cm(.8), Cm(.8)
    sec.different_first_page_header_footer = True
    normal = doc.styles['Normal']
    normal.font.name = '맑은 고딕'
    normal.font.size = Pt(10)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '맑은 고딕')
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.22
    normal.paragraph_format.widow_control = True
    for index, size in ((1, 18), (2, 13), (3, 11)):
        style = doc.styles['Heading ' + str(index)]
        style.font.name = '맑은 고딕'
        style.font.size, style.font.bold = Pt(size), True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '맑은 고딕')
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.page_break_before = False
    for name in ('Record', 'RecordTitle', 'RecordDetail', 'Command', 'CommandSpacer', 'ManualList'):
        if name not in doc.styles:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style = doc.styles[name]
        style.base_style = normal
    doc.styles['Record'].paragraph_format.space_after = Pt(7)
    title = doc.styles['RecordTitle']
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(BLUE)
    title.paragraph_format.space_before = Pt(9)
    title.paragraph_format.space_after = Pt(3)
    title.paragraph_format.keep_with_next = True
    detail = doc.styles['RecordDetail']
    detail.paragraph_format.left_indent = Cm(.35)
    detail.paragraph_format.space_after = Pt(4)
    command = doc.styles['Command']
    command.font.name, command.font.size = 'Consolas', Pt(8.3)
    command.paragraph_format.left_indent = Cm(.25)
    command.paragraph_format.right_indent = Cm(.2)
    command.paragraph_format.space_after = Pt(1)
    command.paragraph_format.line_spacing = 1.05
    shade = OxmlElement('w:shd')
    shade.set(qn('w:fill'), 'F0F4F7')
    command._element.get_or_add_pPr().append(shade)
    doc.styles['CommandSpacer'].paragraph_format.space_after = Pt(2)
    doc.styles['CommandSpacer'].font.size = Pt(2)
    doc.styles['ManualList'].paragraph_format.left_indent = Cm(.25)
    header = sec.header.paragraphs[0]
    font(header.add_run('MX3 + HSL  /  로봇 운영 매뉴얼'), size=8, color='5E7182')
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(footer.add_run('2026.11.09   |   '), size=8, color='5E7182')
    field(footer, 'PAGE', '1')
    font(footer.add_run(' / '), size=8, color='5E7182')
    field(footer, 'NUMPAGES', '1')


def cover(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(80)
    font(p.add_run('MX3 + HSL  ·  ROBOT OPERATIONS'), size=12, bold=True, color=BLUE)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(20)
    font(p.add_run('로봇 운영\n매뉴얼'), size=34, bold=True, color=BLUE)
    font(doc.add_paragraph().add_run('역할 지정 · RViz · 경기장 전환\n로봇 추가 · 4대4 모의 경기 · B-Human 이식 내역'), size=15)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(30)
    font(p.add_run('현재 성공 기준'), size=12, bold=True, color=BLUE)
    add_inline(doc, 'robo04 / ROS 2 Humble / practice 3.9×2.6m')
    add_inline(doc, '실제 이동·공 탐색 접근·RViz 표시: 사용자 관찰상 성공')
    add_inline(doc, '실측 위치 정확도·다중 로봇 팀 경기: 다음 단계에서 확인')
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(22)
    font(p.add_run('작성 기준  2026년 11월 9일'), size=10, color='5E7182')


def quickstart(doc):
    doc.add_heading('빠른 실행 안내 — 현재 robo04', 1).paragraph_format.page_break_before = True
    add_inline(doc, '설치가 끝난 현재 구성 기준입니다. 아래 순서를 따르고 실행 창을 유지합니다.')
    items = [
        ('1  터미널 1: 기본 기능 전체', 'bash /home/robo04/incoming/real_robot_20261109/base.sh'),
        ('2  실제 시작 위치에 세운 뒤 물리 START로 기립', None),
        ('3  터미널 3: START 이후 초기 위치 한 번 입력', 'bash /home/robo04/incoming/real_robot_20261109/init_pose.sh'),
        ('4  터미널 2: 보행·킥·경기 행동 실행', 'bash /home/robo04/incoming/real_robot_20261109/start.sh'),
        ('5  터미널 3: 원하는 역할 하나 선택',
         'bash /home/robo04/incoming/real_robot_20261109/role.sh attacker --seconds 10\n'
         'bash /home/robo04/incoming/real_robot_20261109/role.sh defender --seconds 10\n'
         'bash /home/robo04/incoming/real_robot_20261109/role.sh goalkeeper --seconds 15'),
        ('6  터미널 4: X11 전달을 켠 새 SSH 연결에서 RViz',
         'bash /home/robo04/incoming/hsl_view_20261109/start.sh'),
        ('정지 명령 — 터미널 3', 'bash /home/robo04/incoming/real_robot_20261109/role.sh stop'),
    ]
    for label, command in items:
        p = doc.add_paragraph(style='RecordTitle')
        font(p.add_run(label), bold=True)
        if command:
            codeblock(doc, command)
        else:
            add_inline(doc, '우리 골라인 왼쪽 두 T 사이, 중앙보다 왼쪽 약 76cm, 상대 골대 중앙 방향.')
    add_inline(doc, '역할 명령은 세 개를 연속 실행하지 말고 원하는 것 하나를 고릅니다. 이미 이동했다면 시작 좌표를 다시 입력하지 않습니다.')
    add_inline(doc, '완전 종료: 역할 stop → 로봇 지지 → 물리 STOP으로 토크 OFF → 실행 창 Ctrl+C.')


def contents(doc):
    doc.add_page_break()
    p = doc.add_paragraph()
    font(p.add_run('목차'), size=20, bold=True, color=BLUE)
    return doc.add_paragraph('CONTENTS_ANCHOR')


def fill_contents(doc, anchor):
    for index, heading in enumerate(p for p in doc.paragraphs if p.style.name == 'Heading 1'):
        name = 'chapter_' + str(index)
        start, end = OxmlElement('w:bookmarkStart'), OxmlElement('w:bookmarkEnd')
        start.set(qn('w:id'), str(index + 10))
        start.set(qn('w:name'), name)
        end.set(qn('w:id'), str(index + 10))
        heading._p.insert(1 if heading._p.pPr is not None else 0, start)
        heading._p.append(end)
        paragraph = anchor.insert_paragraph_before()
        link = OxmlElement('w:hyperlink')
        link.set(qn('w:anchor'), name)
        run = OxmlElement('w:r')
        value = OxmlElement('w:t')
        value.text = heading.text
        run.append(value)
        link.append(run)
        paragraph._p.append(link)
    anchor._p.getparent().remove(anchor._p)


def fleet(doc):
    doc.add_heading('10. 로봇 8대 설정 기록란', 1).paragraph_format.page_break_before = True
    add_inline(doc, '아래는 제안 배치입니다. robo04 이외의 접속 정보는 실제 기체를 연결하며 채웁니다. 팀35/36도 모의 경기 설정과 일치시킵니다.')
    rows = list(csv.DictReader((INPUT / 'ROBOT_FLEET_TEMPLATE_20261109.csv').open(encoding='utf-8')))
    for row in rows:
        p = doc.add_paragraph(style='RecordTitle')
        font(p.add_run('%s  ·  팀 %s / 선수 %s' % (row['robot_label'], row['team_number'], row['player_number'])), bold=True)
        known = row['robot_label'] == 'robo04'
        for label, value in [
            ('접속', row['ssh_user'] + '@' + row['ip_address'] if known else '사용자 __________  / IP ____________________'),
            ('워크스페이스', row['workspace'] if known else '____________________________________________'),
            ('역할/지도', ('현재 단독 attacker 시험' if known else '경기 계획 auto') + ' / practice'),
            ('ROS', 'DOMAIN_ID=42 / LOCALHOST_ONLY=1 (각 기체 컴퓨터 내부 ROS 격리)'),
            ('초기 위치', '근사 (-1.95, +0.7583333, 0rad)' if known else '실측 x ________ m / y ________ m / yaw ________ rad'),
            ('하드웨어 기록', 'STM32 by-id ____________________ / U2D2 by-id ____________________'),
        ]:
            p = doc.add_paragraph(style='RecordDetail')
            font(p.add_run(label + '：'), bold=True)
            font(p.add_run(value), size=9.5)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    prepare_styles(doc)
    cover(doc)
    quickstart(doc)
    contents_anchor = contents(doc)
    operations = (INPUT / 'ROBOT_OPERATION_MANUAL_KO_20261109.md').read_text(encoding='utf-8')
    operations = operations.split('\n## 9. B-Human에서 가져온 부분')[0]
    operations = '## ' + operations.split('\n## ', 1)[1]
    markdown(doc, operations)
    doc.add_heading('9. B-Human에서 가져온 기능과 근거', 1).paragraph_format.page_break_before = True
    audit = (INPUT / 'BHUMAN_PORT_AUDIT_KO_20261109.md').read_text(encoding='utf-8')
    markdown(doc, audit, mode='audit')
    fleet(doc)
    fill_contents(doc, contents_anchor)
    doc.core_properties.title = 'MX3 + HSL 로봇 운영 매뉴얼'
    doc.core_properties.subject = '실물 운영, 역할, RViz, 지도 전환, 4대4, B-Human 이식 내역'
    doc.core_properties.author = 'MX3 + HSL 프로젝트'
    doc.core_properties.language = 'ko-KR'
    target = OUTPUT / (NAME + '.docx')
    doc.save(target)
    print('DOCX:', target)
    print('PARAGRAPHS:', len(doc.paragraphs))
    print('NARROW_TABLES:', len(doc.tables))


if __name__ == '__main__':
    main()
