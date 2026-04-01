# Spell & Grammar Corrector (NLP) ✍️

A context-aware text analysis tool that identifies spelling errors and grammatical inconsistencies using Natural Language Processing (NLP). This project features a modern dark-themed GUI and leverages both statistical models and character-level algorithms.

## 🌟 Key Features
- **Contextual Correction**: Uses an N-gram frequency model trained on the `big.txt` corpus (Project Gutenberg).
- **Intelligent Spelling**: Implements the **Levenshtein Distance** algorithm for character-level similarity.
- **Grammar Insights**: Utilizes **NLTK Part-of-Speech (POS) tagging** to detect structural issues.
- **Modern UI**: A sleek, user-friendly interface built with `tkinter` featuring a dark theme.

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/MalaikaAkhtar01/spell-grammar-corrector.git](https://github.com/MalaikaAkhtar01/spell-grammar-corrector.git)
cd spell-grammar-corrector
Install required libraries:

```bash
pip install pyspellchecker nltk
Run the application:

```bash
python spell_grammar_corrector_v2.py
Note: On the first launch, the script will automatically download the necessary NLTK datasets (punkt, brown, words, etc.).

# 📂 Project Structure
spell_grammar_corrector_v2.py: The main Python script containing the NLP logic and GUI.

big.txt: The reference dataset used for training the frequency model.

# 🤝 Contributors
This project was developed as part of an NLP Lab Project by:

Malaika Akhtar (@MalaikaAkhtar01)

Haleema Sadia

Sadia Mazhar

# 📜 Credits
The big.txt dataset is sourced from public domain literature provided by Project Gutenberg.