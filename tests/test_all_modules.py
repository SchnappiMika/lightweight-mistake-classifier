# -*- coding: utf-8 -*-
"""
Comprehensive test suite for all 5 core modules
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} - {detail}')


def test_module1_import():
    print('\n=== Module 1: Multimodal Import ===')
    from src.preprocessing.text_cleaner import TextCleaner
    from src.preprocessing.tokenizer import Tokenizer
    from src.preprocessing.stopwords import StopwordsFilter
    from src.preprocessing.formula import FormulaRecognizer
    from src.preprocessing.image_ocr import ImageOCR
    from src.preprocessing.audio import AudioTranscriber
    check('TextCleaner import', True)
    check('Tokenizer import', True)
    check('StopwordsFilter import', True)
    check('FormulaRecognizer import', True)
    check('ImageOCR import', True)
    check('AudioTranscriber import', True)

    c = TextCleaner()
    check('Text cleaning', c.clean('  1. ??? lim x->0 sinx/x  ') == '??? lim x->0 sinx/x')

    dd = os.path.join(os.getcwd(), 'src/knowledge_base/data/dictionaries')
    Tokenizer.initialize(dd)
    tok = Tokenizer()
    words = tok.segment('???f(x)=x^2???')
    check('jieba tokenization works', len(words) > 3)
    core = tok.extract_core_words(words)
    check('Core word extraction', '??' in core or len(core) > 0)

    sw = StopwordsFilter()
    filtered = sw.filter(['?', '?', '??', '??', '?'])
    check('Stopwords filter', '?' not in filtered and '??' in filtered)


def test_module2_category():
    print('\n=== Module 2: Category Management ===')
    dd = os.path.join(os.getcwd(), 'src/knowledge_base/data/dictionaries')
    if os.path.exists(dd):
        files = [f for f in os.listdir(dd) if f.endswith('.json')]
        check('Subject dictionaries exist', len(files) >= 3, f'Found {len(files)}')
        for fn in files:
            with open(os.path.join(dd, fn), encoding='utf-8') as f:
                data = json.load(f)
                check(f'{fn}: subject field', 'subject' in data)
                check(f'{fn}: has categories', len(data['categories']) > 0)
                for c in data['categories']:
                    check(f'{c["name"]}: has keywords', len(c.get('keywords',[])) > 0)
    else:
        check('Dictionary directory', False)


def test_module3_classify():
    print('\n=== Module 3: Classification Engine ===')
    from src.classification.classifier import Classifier
    clf = Classifier()
    check('Classifier init', clf is not None)

    subjects = clf.get_subjects()
    check('Get subjects', len(subjects) >= 3, f'Got {subjects}')

    cats = clf.get_categories()
    check('Get categories', len(cats) >= 10, f'Got {len(cats)} categories')

    # Test classification
    result = clf.classify_text('??? lim x->0 sinx/x')
    check('Classify text returns result', result is not None)
    check('Classify has final_result', 'final_result' in result)
    cat, score = result['final_result']
    check('Final category not N/A', cat != 'N/A')
    check('Score is positive', score > 0)
    check('Has keyword candidates', len(result['keyword_candidates']) > 0)
    check('Has detected subject', result.get('detected_subject') is not None)

    # Test multiple subjects
    for text, subj in [
        ('??? f(x)=x^2', '\u9ad8\u7b49\u6570\u5b66'),
        ('??????', '\u5927\u5b66\u7269\u7406'),
        ('Read the passage', '\u5927\u5b66\u82f1\u8bed')
    ]:
        r = clf.classify_text(text, subj)
        check(f'Subject filter: {subj}', r['detected_subject'] == subj)

    # Test performance
    sample_texts = [
        '??? lim x->0 sinx/x',
        '???f(x)=x^2?x=2????',
        '????? \u222b_0^1 x^2 dx',
        '???????????0.1m???2s',
        'Read the passage and answer the questions',
        '??????????',
        'Write an essay about environmental protection'
    ]
    t0 = time.time()
    for text in sample_texts:
        clf.classify_text(text)
    elapsed = time.time() - t0
    avg = elapsed / len(sample_texts)
    check(f'Single question under 1s ({avg:.3f}s)', avg < 1.0, f'avg={avg:.3f}s')


def test_module4_feedback():
    print('\n=== Module 4: Feedback & Optimization ===')
    from src.classification.classifier import Classifier
    from src.preprocessing.text_cleaner import TextCleaner
    from src.preprocessing.tokenizer import Tokenizer
    from src.preprocessing.stopwords import StopwordsFilter
    clf = Classifier()

    # Simulate correction
    text = '??? x->0'
    cleaner = TextCleaner(); tok = Tokenizer(); sw = StopwordsFilter()
    cleaned = cleaner.clean(text)
    words = tok.segment(cleaned)
    cw = tok.extract_core_words(words)
    fw = sw.filter(cw, None)

    # Add feedback
    old_count = len(clf.tfidf_matcher.category_texts.get('\u51fd\u6570\u4e0e\u6781\u9650', ''))
    clf.tfidf_matcher.add_sample('\u51fd\u6570\u4e0e\u6781\u9650', ' '.join(fw))
    new_count = len(clf.tfidf_matcher.category_texts.get('\u51fd\u6570\u4e0e\u6781\u9650', ''))
    check('Incremental update works', new_count > old_count,
           f'old={old_count} new={new_count}')


def test_module5_export():
    print('\n=== Module 5: Export ===')
    import pandas as pd
    results = [
        {"text": "test1", "subject": "\u9ad8\u7b49\u6570\u5b66", "category": "\u51fd\u6570\u4e0e\u6781\u9650", "score": 0.95},
        {"text": "test2", "subject": "\u5927\u5b66\u7269\u7406", "category": "\u529b\u5b66", "score": 0.87},
    ]
    df = pd.DataFrame(results)
    check('DataFrame created', len(df) == 2)

    import tempfile
    fp = os.path.join(tempfile.gettempdir(), 'test_export.xlsx')
    with pd.ExcelWriter(fp, engine='openpyxl') as w:
        for subj in df['subject'].unique():
            df[df['subject'] == subj].to_excel(w, sheet_name=str(subj)[:31], index=False)
    check('Excel exported', os.path.exists(fp) and os.path.getsize(fp) > 0)
    os.remove(fp)


def test_gui_modules():
    print('\n=== GUI Modules (import check) ===')
    try:
        from PyQt5.QtWidgets import QApplication
        check('PyQt5 available', True)
    except ImportError:
        check('PyQt5 available', False)
        return

    # Check import without creating QApplication
    for mod in ['import_page', 'classify_page', 'export_page', 'category_page', 'optimize_page']:
        try:
            exec(f'from src.ui.{mod} import {mod[0].upper() + mod[1:].split("_")[0] + "Page"}')
            check(f'{mod} importable', True)
        except Exception as e:
            check(f'{mod} importable', False, str(e))


def run_all():
    print('=' * 55)
    print('  Comprehensive System Test Suite')
    print('=' * 55)

    test_module1_import()
    test_module2_category()
    test_module3_classify()
    test_module4_feedback()
    test_module5_export()
    test_gui_modules()

    total = PASS + FAIL
    print(f'\n{"=" * 55}')
    print(f'  Results: {PASS}/{total} passed, {FAIL}/{total} failed')
    if FAIL == 0:
        print('  All tests passed!')
    else:
        print(f'  {FAIL} test(s) failed')
    print(f'{"=" * 55}')
    return FAIL == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)