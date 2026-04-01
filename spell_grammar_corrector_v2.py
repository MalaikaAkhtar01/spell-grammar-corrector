"""
Context-Based Spelling & Grammatical Error Corrector  — v2
NLP Lab Project

Install dependencies first:
    pip install pyspellchecker nltk

On first run NLTK will auto-download: punkt, averaged_perceptron_tagger, brown, words
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import re
import math
import string
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR THEME
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#1e1e2e"
PANEL_BG   = "#2a2a3e"
ACCENT     = "#7c6af7"
ACCENT2    = "#56c8a0"
TEXT_FG    = "#cdd6f4"
ERROR_FG   = "#f38ba8"
WARN_FG    = "#fab387"
ENTRY_BG   = "#313244"
BORDER     = "#45475a"
BTN_FG     = "#ffffff"

# ─────────────────────────────────────────────────────────────────────────────
#  EDIT DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if c1 == c2 else 1)
            ))
        prev = curr
    return prev[-1]


def generate_edits1(word: str) -> set:
    """All words at edit distance 1."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits  = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:]          for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]      for L, R in splits if R for c in letters]
    inserts    = [L + c + R          for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def generate_edits2(word: str) -> set:
    """All words at edit distance 2."""
    return {e2 for e1 in generate_edits1(word) for e2 in generate_edits1(e1)}


# ─────────────────────────────────────────────────────────────────────────────
#  BIGRAM LANGUAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BigramLM:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams  = defaultdict(Counter)
        self.vocab    = set()
        self.N        = 0

    def train(self, sentences):
        for sent in sentences:
            toks = ['<s>'] + [w.lower() for w in sent] + ['</s>']
            for w in toks:
                self.unigrams[w] += 1
                self.vocab.add(w)
            for a, b in zip(toks, toks[1:]):
                self.bigrams[a][b] += 1
        self.N = sum(self.unigrams.values())

    def prob(self, prev: str, word: str) -> float:
        V = len(self.vocab) or 1
        num = self.bigrams[prev.lower()].get(word.lower(), 0) + 1
        den = self.unigrams.get(prev.lower(), 0) + V
        return num / den

    def score(self, candidate: str, prev: str, nxt: str) -> float:
        s  = math.log(self.prob(prev, candidate) + 1e-10)
        s += math.log(self.prob(candidate, nxt)  + 1e-10)
        return s


# ─────────────────────────────────────────────────────────────────────────────
#  NOISY CHANNEL RANKER
# ─────────────────────────────────────────────────────────────────────────────
def noisy_channel_rank(error: str, candidates: set,
                       lm: BigramLM, prev: str, nxt: str,
                       top_n: int = 6) -> list:
    scored = []
    error_lower = error.lower()
    for cand in candidates:
        dist          = levenshtein(error_lower, cand)
        channel_score = math.log(max(math.exp(-dist), 1e-10))
        lm_score      = lm.score(cand, prev, nxt)
        total         = channel_score + lm_score
        scored.append((cand, total, dist))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
#  SPELL CHECKER  (pyspellchecker + fallback)
# ─────────────────────────────────────────────────────────────────────────────
class SpellChecker:
    def __init__(self):
        self.speller  = None
        self.lm_vocab = set()

    def load(self, lm_vocab: set):
        self.lm_vocab = lm_vocab
        try:
            from spellchecker import SpellChecker as SC
            self.speller = SC()
        except ImportError:
            self.speller = None

    def is_correct(self, word: str) -> bool:
        w = word.lower()
        if not w.isalpha() or len(w) <= 2:
            return True
        if self.speller:
            return len(self.speller.unknown([w])) == 0
        return w in self.lm_vocab

    def candidates(self, word: str) -> set:
        w = word.lower()
        if self.speller:
            cands = self.speller.candidates(w) or set()
        else:
            cands = set()

        # augment with edit-distance candidates that are in lm_vocab
        e1 = generate_edits1(w) & self.lm_vocab
        e2 = generate_edits2(w) & self.lm_vocab
        cands = cands | e1 | e2

        # remove the original misspelling
        cands.discard(w)
        return cands


# ─────────────────────────────────────────────────────────────────────────────
#  GRAMMAR CHECKER  (comprehensive rule-based)
# ─────────────────────────────────────────────────────────────────────────────
class GrammarChecker:

    # Common irregular verb forms: {wrong: correct}
    IRREGULAR_VERBS = {
        "goed":"went","catched":"caught","runned":"ran","thinked":"thought",
        "buyed":"bought","bringed":"brought","teached":"taught","speaked":"spoke",
        "writed":"wrote","drived":"drove","taked":"took","gived":"gave",
        "feeled":"felt","knowed":"knew","selled":"sold","telled":"told",
        "comed":"came","becomed":"became","breaked":"broke","choosed":"chose",
        "falled":"fell","forgotted":"forgot","growed":"grew","holded":"held",
        "hurted":"hurt","keeped":"kept","leaved":"left","lended":"lent",
        "loosed":"lost","meeted":"met","payed":"paid","readed":"read",
        "ringed":"rang","rised":"rose","sitted":"sat","sleeped":"slept",
        "standed":"stood","swimmed":"swam","throwd":"threw","wored":"wore",
    }

    # Wrong contraction / common word confusion
    WORD_CONFUSIONS = {
        "dont":    "don't",  "cant":   "can't",  "wont":   "won't",
        "isnt":    "isn't",  "arent":  "aren't",  "wasnt":  "wasn't",
        "werent":  "weren't","doesnt": "doesn't", "didnt":  "didn't",
        "hasnt":   "hasn't", "havent": "haven't", "hadnt":  "hadn't",
        "shouldnt":"shouldn't","wouldnt":"wouldn't","couldnt":"couldn't",
        "its":     None,     # context-dependent, flag only
        "their":   None,
        "there":   None,
        "they're": None,
        "your":    None,
        "you're":  None,
        "alot":    "a lot",
        "alright": "all right",
        "irregardless": "regardless",
        "could of":  "could have",
        "would of":  "would have",
        "should of": "should have",
    }

    DOUBLE_NEGATIVES = {
        ("not","nothing"), ("not","nobody"), ("not","nowhere"),
        ("not","never"),   ("never","nothing"), ("no","nothing"),
        ("n't","nothing"), ("n't","nobody"),    ("n't","nowhere"),
        ("n't","never"),
    }

    # Subject → correct auxiliary
    SV_RULES = {
        # (subject, wrong_verb) : suggestion
        ("i","are"):    "I am",        ("i","is"):     "I am",
        ("he","are"):   "he is",       ("she","are"):  "she is",
        ("it","are"):   "it is",
        ("they","is"):  "they are",    ("they","was"): "they were",
        ("we","is"):    "we are",      ("we","was"):   "we were",
        ("you","is"):   "you are",     ("you","was"):  "you were",
        ("he","were"):  "he was",      ("she","were"): "she was",
        ("it","were"):  "it was",
        ("they","has"): "they have",   ("he","have"):  "he has",
        ("she","have"): "she has",     ("it","have"):  "it has",
        ("i","has"):    "I have",
    }

    def check(self, tokens: list) -> list:
        """Returns list of {index, word, message, suggestion}."""
        errors = []
        low    = [t.lower() for t in tokens]
        n      = len(tokens)

        for i, (tok, lo) in enumerate(zip(tokens, low)):
            if not tok.isalpha():
                continue

            # 1. Irregular verb forms
            if lo in self.IRREGULAR_VERBS:
                errors.append({
                    'index':      i,
                    'word':       tok,
                    'message':    f'Wrong verb form: "{tok}"',
                    'suggestion': self.IRREGULAR_VERBS[lo],
                    'type':       'grammar'
                })

            # 2. Missing apostrophe contractions
            if lo in ("dont","cant","wont","isnt","arent","wasnt","werent",
                      "doesnt","didnt","hasnt","havent","hadnt",
                      "shouldnt","wouldnt","couldnt"):
                correct = self.WORD_CONFUSIONS.get(lo, lo)
                errors.append({
                    'index':      i,
                    'word':       tok,
                    'message':    f'Missing apostrophe: "{tok}"',
                    'suggestion': correct,
                    'type':       'grammar'
                })

            # 3. "alot" → "a lot"
            if lo == "alot":
                errors.append({
                    'index':      i,
                    'word':       tok,
                    'message':    '"alot" is not a word',
                    'suggestion': 'a lot',
                    'type':       'grammar'
                })

            # 4. Article a/an
            if lo in ('a', 'an') and i + 1 < n and tokens[i + 1].isalpha():
                next_w = tokens[i + 1].lower()
                starts_vowel = next_w[0] in 'aeiou'
                if lo == 'a' and starts_vowel:
                    errors.append({
                        'index':      i,
                        'word':       tok,
                        'message':    f'Use "an" before "{tokens[i+1]}"',
                        'suggestion': 'an',
                        'type':       'grammar'
                    })
                elif lo == 'an' and not starts_vowel:
                    errors.append({
                        'index':      i,
                        'word':       tok,
                        'message':    f'Use "a" before "{tokens[i+1]}"',
                        'suggestion': 'a',
                        'type':       'grammar'
                    })

            # 5. Repeated word
            if i > 0 and lo == low[i - 1] and tok.isalpha():
                errors.append({
                    'index':      i,
                    'word':       tok,
                    'message':    f'Repeated word: "{tok}"',
                    'suggestion': '',
                    'type':       'grammar'
                })

        # 6. Subject-verb agreement (window of 2)
        for i in range(n - 1):
            pair = (low[i], low[i + 1])
            if pair in self.SV_RULES:
                errors.append({
                    'index':      i + 1,
                    'word':       tokens[i + 1],
                    'message':    f'S-V disagreement: "{tokens[i]} {tokens[i+1]}"',
                    'suggestion': self.SV_RULES[pair],
                    'type':       'grammar'
                })

        # 7. Double negation (window scan)
        for i in range(n - 1):
            for j in range(i + 1, min(i + 5, n)):
                if (low[i], low[j]) in self.DOUBLE_NEGATIVES:
                    errors.append({
                        'index':      j,
                        'word':       tokens[j],
                        'message':    f'Double negation: "{tokens[i]}" + "{tokens[j]}"',
                        'suggestion': '',
                        'type':       'grammar'
                    })
                    break

        # 8. "could/would/should of" → "have"
        for i in range(n - 1):
            if low[i] in ('could','would','should') and low[i+1] == 'of':
                errors.append({
                    'index':      i + 1,
                    'word':       tokens[i + 1],
                    'message':    f'"{tokens[i]} of" should be "{tokens[i]} have"',
                    'suggestion': 'have',
                    'type':       'grammar'
                })

        # deduplicate by index (keep first occurrence)
        seen   = set()
        unique = []
        for e in errors:
            if e['index'] not in seen:
                seen.add(e['index'])
                unique.append(e)

        return unique


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CORRECTOR
# ─────────────────────────────────────────────────────────────────────────────
class Corrector:
    def __init__(self, on_status=None):
        self.lm      = BigramLM()
        self.spell   = SpellChecker()
        self.grammar = GrammarChecker()
        self.ready   = False
        self.log     = on_status or print

    # ── corpus loading ────────────────────────────────────────────────
    def load(self):
        try:
            import nltk
            for pkg in ['punkt','averaged_perceptron_tagger',
                        'brown','words','punkt_tab',
                        'averaged_perceptron_tagger_eng']:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

            self.log("Training bigram model …")
            from nltk.corpus import brown
            sents = list(brown.sents())
            self.lm.train(sents)

            self.log("Loading vocabulary …")
            from nltk.corpus import words as w
            vocab = set(x.lower() for x in w.words() if x.isalpha())
            self.lm.vocab.update(vocab)
            self.spell.load(self.lm.vocab)

        except Exception as ex:
            self.log(f"NLTK unavailable ({ex}), using built-in corpus …")
            self._mini_corpus()

        self.ready = True
        self.log("Ready ✓")

    def _mini_corpus(self):
        raw = """
        the cat sat on the mat . the dog ran in the park .
        she is going to the store to buy some apples and oranges .
        he does not know anything about natural language processing .
        they were happy when the results came out yesterday .
        i have been studying for the exam since morning .
        the weather is very nice today and i am feeling great .
        she received a letter from her friend last week .
        spelling errors are common in written text and must be corrected .
        edit distance measures how similar two strings are to each other .
        bigram models use the previous word to predict the next word in a sentence .
        the quick brown fox jumps over the lazy dog near the river .
        we should always check our work before submitting it to the teacher .
        natural language processing helps computers understand and generate human text .
        i believe that this project will get very good marks in the lab .
        the students are working hard to complete their assignments on time .
        he can not find his book anywhere in the house or the library .
        they did not know what to do when the power went out last night .
        she has never been to paris but she would love to visit it someday .
        the government announced new policies for education and health care .
        people should not use double negatives in formal writing or speech .
        """.strip().split('\n')
        sents = [line.strip().split() for line in raw if line.strip()]
        self.lm.train(sents)
        vocab = {w for s in sents for w in s if w.isalpha()}
        # add a broad common-word list
        common = (
            "able about above across after again age ago agree air all allow "
            "almost alone along already also although always among and animal "
            "another answer any anyone anything anyway area around ask at away "
            "back bad ball bank base be because become before begin being believe "
            "below best better big body book both boy bring build but buy by call "
            "came can care carry case cat cause change child city class clean "
            "clear close come common complete control could country cut day dead "
            "dear decide deep did different difficult do done door down draw dream "
            "drive drop dry during each early earth easy eat either end enough "
            "even ever every example eye face fact fall family far fast feel few "
            "field fight figure find fire first follow food force form friend from "
            "front full further get give go good great ground group grow had half "
            "hand hard have he head hear heart help her here high him his hold "
            "home hour house how human idea if important in inside instead it "
           "job just keep kind know land large last late laugh lead learn leave "
            "left let life light like line list listen little live long look love "
            "low made make man many mark matter may me mean meet mile mind money "
            "more most move much music must my name near need never new next "
            "night no none nor not now number of off often old on once only open "
            "or order other our out over own page paper part past people place "
            "plan play point poor power press problem pull push put question "
            "quite reach read ready real reason record remain remember rest "
            "right road rock room round rule run same say school sea second "
            "seem seen self send set she short show side simple since small so "
            "some soon sound space speak special stand start state stay still "
            "stop story street strong study such sun sure system take talk "
            "tell than that the their them then there these they thing think "
            "this those though thought time to today together too top toward "
            "town tree true try turn under until up us use usually very visit "
            "walk want was watch water way we well went were what when where "
            "which while white who whole why wide will wish with without word "
            "work world would write year yes yet you young your "
            "bought brought caught chose drove felt flew forgot gave grew held "
            "kept knew led left lent lost met paid rang rang rose sat slept "
            "sold spoke stood swam threw wore went thought understood won wrote "
            "been done gone seen become come run begun bitten broken chosen "
            "eaten fallen gotten given hidden known mistaken ridden risen "
            "shaken stolen taken woken written "
            "beautiful interesting important different necessary government "
            "experience knowledge understand technology information education "
            "university language sentence grammar spelling mistake error "
            "correct correction suggestion analysis paragraph document "
        ).split()
        vocab.update(common)
        self.lm.vocab.update(vocab)
        self.spell.load(self.lm.vocab)

    # ── main pipeline ─────────────────────────────────────────────────
    def analyse(self, text: str) -> dict:
        if not self.ready:
            return {}

        # tokenise — keep punctuation as separate tokens
        tokens = re.findall(r"\b\w[\w']*\b|[^\w\s]", text)

        # ── spelling ──────────────────────────────────────────────────
        spell_errors = {}   # index → [(suggestion, score, dist), ...]
        for i, tok in enumerate(tokens):
            if not tok.isalpha() or len(tok) <= 2:
                continue
            if self.spell.is_correct(tok):
                continue

            prev = tokens[i - 1] if i > 0     else '<s>'
            nxt  = tokens[i + 1] if i < len(tokens) - 1 else '</s>'

            cands = self.spell.candidates(tok)
            if not cands:
                continue

            ranked = noisy_channel_rank(tok, cands, self.lm, prev, nxt, top_n=6)
            spell_errors[i] = ranked   # [(word, score, dist), ...]

        # ── grammar ───────────────────────────────────────────────────
        gram_errors = self.grammar.check(tokens)

        # ── corrected text ────────────────────────────────────────────
        corrected = list(tokens)
        for idx, ranked in spell_errors.items():
            if ranked:
                corrected[idx] = ranked[0][0]   # best spelling fix

        for err in gram_errors:
            if err.get('suggestion'):
                corrected[err['index']] = err['suggestion']

        corrected_text = self._rejoin(corrected)

        return {
            'tokens':          tokens,
            'spell_errors':    spell_errors,
            'gram_errors':     gram_errors,
            'corrected_text':  corrected_text,
        }

    @staticmethod
    def _rejoin(tokens):
        out = ''
        for i, tok in enumerate(tokens):
            if tok in string.punctuation and tok not in '([{"\'':
                out += tok
            elif i == 0:
                out += tok
            else:
                out += ' ' + tok
        return out.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Context-Based Spelling & Grammar Corrector  —  NLP Lab v2")
        self.geometry("1150x740")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        self.corrector = Corrector(on_status=self._set_status)
        self.result    = {}

        self._style()
        self._build()
        threading.Thread(target=self.corrector.load, daemon=True).start()

    # ── ttk styles ────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure("TNotebook",       background=DARK_BG, borderwidth=0)
        s.configure("TNotebook.Tab",   background=PANEL_BG, foreground=TEXT_FG,
                    padding=[14, 7],   font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#fff")])
        s.configure("Treeview",
                    background=PANEL_BG, foreground=TEXT_FG,
                    fieldbackground=PANEL_BG, rowheight=30,
                    font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
                    background=ACCENT, foreground="#fff",
                    font=("Segoe UI", 10, "bold"))
        s.map("Treeview", background=[("selected", ACCENT)])

    # ── layout ────────────────────────────────────────────────────────
    def _build(self):
        # top accent bar
        tk.Frame(self, bg=ACCENT, height=4).pack(fill='x')

        # header
        hdr = tk.Frame(self, bg=DARK_BG, pady=8)
        hdr.pack(fill='x', padx=18)
        tk.Label(hdr, text="🔤  Context-Based Spelling & Grammar Corrector",
                 bg=DARK_BG, fg=TEXT_FG,
                 font=("Segoe UI", 14, "bold")).pack(side='left')
        self._status_var = tk.StringVar(value="Loading …")
        tk.Label(hdr, textvariable=self._status_var,
                 bg=DARK_BG, fg=ACCENT2,
                 font=("Segoe UI", 10)).pack(side='right')

        # paned window
        pw = tk.PanedWindow(self, orient='horizontal', bg=DARK_BG,
                            sashwidth=6, sashrelief='flat')
        pw.pack(fill='both', expand=True, padx=12, pady=(0, 10))
        pw.add(self._left_panel(pw),  minsize=460)
        pw.add(self._right_panel(pw), minsize=360)

        # bottom bar
        bar = tk.Frame(self, bg=PANEL_BG, height=28)
        bar.pack(fill='x', side='bottom')
        self._info_var = tk.StringVar(value="Enter or paste text, then click Analyse.")
        tk.Label(bar, textvariable=self._info_var,
                 bg=PANEL_BG, fg=BORDER, font=("Segoe UI", 9), anchor='w'
                 ).pack(side='left', padx=10)
        tk.Label(bar, text="NLP Lab  •  Edit Distance · Bigrams · Noisy Channel",
                 bg=PANEL_BG, fg=BORDER, font=("Segoe UI", 9)
                 ).pack(side='right', padx=10)

    def _left_panel(self, parent):
        f = tk.Frame(parent, bg=DARK_BG)

        tk.Label(f, text="📝  Input Text", bg=DARK_BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor='w', pady=(6, 3))

        self._input = scrolledtext.ScrolledText(
            f, wrap='word', height=9, bg=ENTRY_BG, fg=TEXT_FG,
            insertbackground=TEXT_FG, font=("Segoe UI", 12),
            relief='flat', borderwidth=1,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, padx=10, pady=8)
        self._input.pack(fill='x')
        self._input.insert('1.0',
            "She dont knows nothing about the grammer mistakes. "
            "Their are a many erors in this sentance and we ned to fix them. "
            "He goed to the store and buyed a apple.")

        # buttons
        br = tk.Frame(f, bg=DARK_BG, pady=8)
        br.pack(fill='x')
        self._btn(br, "🔍  Analyse",      self._analyse,       ACCENT ).pack(side='left', padx=(0,8))
        self._btn(br, "✅  Auto-Correct",  self._auto_correct,  ACCENT2).pack(side='left', padx=(0,8))
        self._btn(br, "🗑  Clear",         self._clear,        "#585b70").pack(side='left')

        # output
        tk.Label(f, text="📋  Highlighted Output  (🔴 Spelling  🟠 Grammar)",
                 bg=DARK_BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor='w', pady=(6, 3))

        self._output = scrolledtext.ScrolledText(
            f, wrap='word', height=11, bg=ENTRY_BG, fg=TEXT_FG,
            insertbackground=TEXT_FG, font=("Segoe UI", 12),
            state='disabled', relief='flat', borderwidth=1,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, padx=10, pady=8)
        self._output.pack(fill='both', expand=True)

        self._output.tag_config('spell',   foreground=ERROR_FG,
                                underline=True, font=("Segoe UI", 12, "bold"))
        self._output.tag_config('grammar', foreground=WARN_FG,
                                underline=True, font=("Segoe UI", 12, "bold"))
        self._output.tag_config('ok',      foreground=TEXT_FG,
                                font=("Segoe UI", 12))
        return f

    def _right_panel(self, parent):
        f  = tk.Frame(parent, bg=DARK_BG)
        nb = ttk.Notebook(f)
        nb.pack(fill='both', expand=True)

        # Tab 1 — Spelling
        st = tk.Frame(nb, bg=DARK_BG)
        nb.add(st, text="🔴  Spelling")
        self._spell_tab(st)

        # Tab 2 — Grammar
        gt = tk.Frame(nb, bg=DARK_BG)
        nb.add(gt, text="🟠  Grammar")
        self._gram_tab(gt)

        # Tab 3 — Corrected
        ct = tk.Frame(nb, bg=DARK_BG)
        nb.add(ct, text="✅  Corrected")
        self._corrected_tab(ct)

        return f

    def _spell_tab(self, parent):
        tk.Label(parent, text="Double-click a row to apply that correction",
                 bg=DARK_BG, fg=BORDER, font=("Segoe UI", 9)
                 ).pack(anchor='w', padx=8, pady=(6, 2))

        c  = tk.Frame(parent, bg=DARK_BG)
        c.pack(fill='both', expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(c); sb.pack(side='right', fill='y')

        self._spell_tree = ttk.Treeview(
            c, columns=('word','best','others'),
            show='headings', yscrollcommand=sb.set)
        sb.config(command=self._spell_tree.yview)
        self._spell_tree.heading('word',   text='Misspelled')
        self._spell_tree.heading('best',   text='Best Fix')
        self._spell_tree.heading('others', text='Other Suggestions')
        self._spell_tree.column('word',   width=100, anchor='center')
        self._spell_tree.column('best',   width=100, anchor='center')
        self._spell_tree.column('others', width=230, anchor='w')
        self._spell_tree.pack(fill='both', expand=True)
        self._spell_tree.bind('<Double-1>', self._apply_spell)

    def _gram_tab(self, parent):
        tk.Label(parent, text="Grammar issues found in your text",
                 bg=DARK_BG, fg=BORDER, font=("Segoe UI", 9)
                 ).pack(anchor='w', padx=8, pady=(6, 2))

        c  = tk.Frame(parent, bg=DARK_BG)
        c.pack(fill='both', expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(c); sb.pack(side='right', fill='y')

        self._gram_tree = ttk.Treeview(
            c, columns=('word','issue','fix'),
            show='headings', yscrollcommand=sb.set)
        sb.config(command=self._gram_tree.yview)
        self._gram_tree.heading('word',  text='Token')
        self._gram_tree.heading('issue', text='Issue')
        self._gram_tree.heading('fix',   text='Suggestion')
        self._gram_tree.column('word',  width=90,  anchor='center')
        self._gram_tree.column('issue', width=230, anchor='w')
        self._gram_tree.column('fix',   width=110, anchor='center')
        self._gram_tree.pack(fill='both', expand=True)

    def _corrected_tab(self, parent):
        tk.Label(parent, text="Auto-corrected version of your text",
                 bg=DARK_BG, fg=BORDER, font=("Segoe UI", 9)
                 ).pack(anchor='w', padx=8, pady=(6, 2))

        self._corrected_box = scrolledtext.ScrolledText(
            parent, wrap='word', bg=ENTRY_BG, fg=ACCENT2,
            font=("Segoe UI", 12), relief='flat', state='disabled',
            padx=10, pady=8)
        self._corrected_box.pack(fill='both', expand=True, padx=8, pady=4)

        copy_btn = self._btn(parent, "📋  Copy to Input", self._copy_corrected, ACCENT)
        copy_btn.pack(pady=(0, 8))

    # ── widget helper ─────────────────────────────────────────────────
    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=BTN_FG, activebackground=color,
                         relief='flat', padx=14, pady=6,
                         font=("Segoe UI", 10, "bold"), cursor='hand2')

    # ── status ────────────────────────────────────────────────────────
    def _set_status(self, msg):
        self.after(0, lambda: self._status_var.set(msg))

    # ── actions ───────────────────────────────────────────────────────
    def _analyse(self):
        if not self.corrector.ready:
            messagebox.showinfo("Please wait", "Model still loading, try again shortly.")
            return
        text = self._input.get('1.0', 'end').strip()
        if not text:
            return
        self._set_status("Analysing …")
        self.result = self.corrector.analyse(text)
        self._render_output()
        self._fill_spell_tab()
        self._fill_gram_tab()
        self._fill_corrected_tab()
        ns = len(self.result.get('spell_errors', {}))
        ng = len(self.result.get('gram_errors',  []))
        self._info_var.set(f"{ns} spelling error(s)  •  {ng} grammar issue(s) found.")
        self._set_status("Done ✓")

    def _render_output(self):
        tokens      = self.result.get('tokens', [])
        spell_idxs  = set(self.result.get('spell_errors', {}).keys())
        gram_idxs   = {e['index'] for e in self.result.get('gram_errors', [])}

        self._output.config(state='normal')
        self._output.delete('1.0', 'end')

        for i, tok in enumerate(tokens):
            if i in spell_idxs:
                tag = 'spell'
            elif i in gram_idxs:
                tag = 'grammar'
            else:
                tag = 'ok'

            if tok in string.punctuation and tok not in '([{"\'':
                self._output.insert('end', tok, tag)
            elif i == 0:
                self._output.insert('end', tok, tag)
            else:
                self._output.insert('end', ' ' + tok, tag)

        self._output.config(state='disabled')

    def _fill_spell_tab(self):
        for r in self._spell_tree.get_children():
            self._spell_tree.delete(r)
        for idx, ranked in self.result.get('spell_errors', {}).items():
            word   = self.result['tokens'][idx]
            best   = ranked[0][0] if ranked else '—'
            others = ',  '.join(r[0] for r in ranked[1:5])
            self._spell_tree.insert('', 'end', iid=str(idx),
                                    values=(word, best, others))

    def _fill_gram_tab(self):
        for r in self._gram_tree.get_children():
            self._gram_tree.delete(r)
        for err in self.result.get('gram_errors', []):
            self._gram_tree.insert('', 'end',
                values=(err['word'], err['message'], err.get('suggestion','—')))

    def _fill_corrected_tab(self):
        corrected = self.result.get('corrected_text', '')
        self._corrected_box.config(state='normal')
        self._corrected_box.delete('1.0', 'end')
        self._corrected_box.insert('1.0', corrected)
        self._corrected_box.config(state='disabled')

    def _apply_spell(self, _event):
        sel = self._spell_tree.selection()
        if not sel:
            return
        idx    = int(sel[0])
        ranked = self.result.get('spell_errors', {}).get(idx, [])
        if not ranked:
            return
        best   = ranked[0][0]
        tokens = self.result['tokens']
        old    = tokens[idx]
        tokens[idx] = best
        rebuilt = self.corrector._rejoin(tokens)
        self._input.delete('1.0', 'end')
        self._input.insert('1.0', rebuilt)
        self._info_var.set(f'Applied: "{old}" → "{best}". Click Analyse to refresh.')

    def _auto_correct(self):
        if not self.corrector.ready:
            messagebox.showinfo("Please wait", "Model still loading.")
            return
        text = self._input.get('1.0', 'end').strip()
        if not text:
            return
        self.result = self.corrector.analyse(text)
        corrected   = self.result.get('corrected_text', text)
        self._input.delete('1.0', 'end')
        self._input.insert('1.0', corrected)
        self._analyse()

    def _copy_corrected(self):
        corrected = self.result.get('corrected_text', '')
        if corrected:
            self._input.delete('1.0', 'end')
            self._input.insert('1.0', corrected)
            self._info_var.set("Corrected text copied to input. Click Analyse to verify.")

    def _clear(self):
        self._input.delete('1.0', 'end')
        self._output.config(state='normal')
        self._output.delete('1.0', 'end')
        self._output.config(state='disabled')
        for t in (self._spell_tree, self._gram_tree):
            for r in t.get_children():
                t.delete(r)
        self._corrected_box.config(state='normal')
        self._corrected_box.delete('1.0', 'end')
        self._corrected_box.config(state='disabled')
        self.result = {}
        self._info_var.set("Cleared.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
