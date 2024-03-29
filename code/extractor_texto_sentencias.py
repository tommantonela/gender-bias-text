import fitz
import re
from collections import deque
from collections import Counter
import pandas as pd
import numpy as np
import os

def get_pdf_elements(filename):
    annotations = deque()
    text_lines = deque()
    page_count = None
    with fitz.open(filename) as doc:
        page_count = doc.page_count
        for i in range(0,doc.page_count):
            page = doc[i]
            for b in page.get_text("dict")["blocks"]:
                if b['type'] == 0: # bloque de texto

                    text = []
                    bbox = []
                    for x in b['lines']:
                        for y in x['spans']:
                            text.append(y['text'])
                            bbox.append(y['bbox'])
                    text_lines.append({'text':re.sub('\s+',' ',' '.join(text)),'page':i,'bbox':(bbox[0][0],bbox[0][1],bbox[-1][2],bbox[-1][3])})
 
            wlist = page.get_text('words')
            for annot in page.annots():
                if annot.type[0] in (8, 9, 10, 11): # 0 son los comentarios
                    annotations.append({'page':i,'text':_extract_annot(annot,wlist).strip()})

    i = 1 # esto es para capturar las líneas que quedaron separadas, pero están en la misma línea "física"
    line = []
    bbox = []
    lines = []
    line.append(text_lines[0]['text'])
    bbox.append(text_lines[0]['bbox'])
    while i < len(text_lines):
        first_page = text_lines[i]['page']
        while i < len(text_lines) and text_lines[i]['bbox'][1] == text_lines[i-1]['bbox'][1] and text_lines[i]['bbox'][3] == text_lines[i-1]['bbox'][3]:
            line.append(text_lines[i]['text'])
            bbox.append(text_lines[i]['bbox'])
            i += 1
        if len(line) > 0:
            lines.append({'text':re.sub('\s+',' ',' '.join(line)),'page':first_page,'bbox':(bbox[0][0],bbox[0][1],bbox[-1][2],bbox[-1][3])})
            line.clear()
            bbox.clear()
        if i < len(text_lines):
            line.append(text_lines[i]['text'])
            bbox.append(text_lines[i]['bbox'])
        i += 1
    if len(line) > 0:
        lines.append({'text':re.sub('\s+',' ',' '.join(line)),'page':first_page,'bbox':(bbox[0][0],bbox[0][1],bbox[-1][2],bbox[-1][3])})

    return lines,page_count,annotations

def prune_lines(text_lines,num_pages): # si aparece tantas veces como páginas tiene o la mitad de las páginas, se borra
    counts = Counter()
    for p in text_lines:
        counts[p['text']] += 1
    to_remove = set(k for k,v in counts.items() if v in range(num_pages-2,num_pages+2) or v in range(int(num_pages/2)-1,int(num_pages/2)+2))
    return [x for x in text_lines if x['text'] not in to_remove]

def divide_paragraphs(text_lines):
    paragraphs = deque()
    i = 0
    paragraph = deque()
    while i < len(text_lines):
        first_page = text_lines[i]['page']
        while i < len(text_lines) and re.search('(\.|\."|\.”|\.\-|\.\/\/\-|\.\-)\s?$',text_lines[i]['text']) is None: # ciclo hasta que encuentre un punto
            paragraph.append(text_lines[i]['text'])
            i += 1
        if i < len(text_lines):
            paragraph.append(text_lines[i]['text'])

            if not text_lines[i]['text'].endswith(' ') or text_lines[i]['text'].endswith('.-') or text_lines[i]['text'].endswith('.- ') \
            or text_lines[i]['text'].endswith('.//-') or text_lines[i]['text'].endswith('.//- '):

                tt = re.sub('\s+',' ',' '.join(paragraph)).replace('- ','')
                if len(tt) > 1:
                    paragraphs.append({'page':first_page,'text':tt})
                paragraph.clear()

            elif i+1 < len(text_lines):
                if text_lines[i]['bbox'][0] >= text_lines[i+1]['bbox'][0]-2 and text_lines[i]['bbox'][0] <= text_lines[i+1]['bbox'][0]+2:
                    if text_lines[i]['bbox'][2] >= text_lines[i+1]['bbox'][2]-2 and text_lines[i]['bbox'][2] <= text_lines[i+1]['bbox'][2]+2:
#   
                        pass
    
                    elif int(text_lines[i]['bbox'][2]-text_lines[i]['bbox'][0]) in range (int(text_lines[i+1]['bbox'][2]-text_lines[i+1]['bbox'][0])-1,int(text_lines[i+1]['bbox'][2]-text_lines[i+1]['bbox'][0])+1):

                        pass
                    elif int(text_lines[i]['bbox'][2]-text_lines[i]['bbox'][0]) > int(text_lines[i+1]['bbox'][2]-text_lines[i+1]['bbox'][0]):
#                   
                        pass
                    else: # ver si esto no rompe el resto de las cosas!
                        tt = re.sub('\s+',' ',' '.join(paragraph)).replace('- ','')
                        if len(tt) > 1:
                            paragraphs.append({'page':first_page,'text':tt})
                        paragraph.clear()
                elif int(text_lines[i]['bbox'][2] - text_lines[i]['bbox'][0]) in range(int(text_lines[i+1]['bbox'][2] - text_lines[i+1]['bbox'][0])-1,int(text_lines[i+1]['bbox'][2] - text_lines[i+1]['bbox'][0])+1):
                     pass
                elif text_lines[i]['bbox'][2] <= text_lines[i+1]['bbox'][2]-1 or text_lines[i]['bbox'][2] >= text_lines[i+1]['bbox'][2]+1:
                    tt = re.sub('\s+',' ',' '.join(paragraph)).replace('- ','')
                    if len(tt) > 1:
                        paragraphs.append({'page':first_page,'text':tt})
                    paragraph.clear()
            else: # nothing to do, last line
                tt = re.sub('\s+',' ',' '.join(paragraph)).replace('- ','')
                if len(tt) > 1:
                    paragraphs.append({'page':first_page,'text':tt})
                paragraph.clear()
        else:
            tt = re.sub('\s+',' ',' '.join(paragraph)).replace('- ','')
            if len(tt) > 1:
                paragraphs.append({'page':first_page,'text':tt})
            paragraph.clear()
        i += 1
    return paragraphs

def _check_contain(r_word, points,_threshold_intersection=0.7):
    r = fitz.Quad(points).rect
    r.intersect(r_word)
    if r.get_area() >= r_word.get_area() * _threshold_intersection:
        contain = True
    else:
        contain = False
    return contain


def _extract_annot(annot, words_on_page):
    quad_points = annot.vertices
    quad_count = int(len(quad_points) / 4)
    sentences = []
    for i in range(quad_count):
        points = quad_points[i * 4: i * 4 + 4]
        words = [
            w for w in words_on_page if
            _check_contain(fitz.Rect(w[:4]), points)
        ]
        sentences.append(' '.join(w[4] for w in words).replace('- ',''))
    sentence = ' '.join(sentences).replace('- ','')
    sentence = re.sub('^-','',sentence)
    sentence = re.sub('-$','',sentence)

    return sentence

# ambas estructuras tienen número de página, se puede limitar la búsqueda
def match_annotations(paragraphs,annotations):
    paragraphs = divide_paragraphs(prune_lines(text_lines,page_count))
    last_max = 0
    for annot in annotations:
        annot_list = annot['text'].strip().split(' ')
        while len(annot_list) > 0:

            max_ = None
            for j in range(max(last_max-6,0),len(paragraphs)):
                i = 0
                while i < len(annot_list) and ' '.join(annot_list[0:i]).strip() in paragraphs[j]['text']:
                    i += 1
                if (i == 1 and len(annot_list) == 1 and annot_list[0] in paragraphs[j]['text']) or (i > 1 and ' '.join(annot_list[0:i-1]).strip() in paragraphs[j]['text']):
                    if max_ is None or i > max_[1]:
                        max_ = (j,i)
            if 'highlight' not in paragraphs[max_[0]]:
                paragraphs[max_[0]]['highlight'] = deque()
            paragraphs[max_[0]]['highlight'].append(' '.join(annot_list[0:max_[1]]))
            annot_list = annot_list[max_[1]:]
            last_max = max_[0]
    return paragraphs

def parse_pdf(filename):
    text_lines, page_count, annotations = get_pdf_elements(filename)
    text_lines = prune_lines(text_lines,page_count)
    paragraphs = divide_paragraphs(text_lines)
    paragraphs = match_annotations(paragraphs,annotations)
    return paragraphs

# Iterar por los documentos del directorio y hacer el df para guardar
dir_path = './'
listi = deque()
for file in os.listdir(dir_path):
    if not file.endswith('pdf'):
        continue
    print(file)
    paragraphs = parse_pdf(dir_path + filename)
    for p in paragraphs:
        p['filename'] = filename
    listi.extend(paragraphs)

df = pd.DataFrame(listi)
df.to_pickle('df_sentencias.pickle')
df
