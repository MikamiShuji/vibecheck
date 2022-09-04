import string
import itertools
from wiki_ru_wordnet import WikiWordnet
from typing import Tuple, Optional

from stanza.pipeline.core import Pipeline, DownloadMethod, Document
from stanza.models.common.doc import Word
from pandas import DataFrame as df
from silero import silero_te

from vibe_check.constants import processor_list, greeting_keywords, parting_keywords


class InsightAnalyzer:

    def __init__(self):
        self.nlp = Pipeline('ru', processors=processor_list,
                            download_method=DownloadMethod.REUSE_RESOURCES)
        _, _, _, _, self.preprocessor = silero_te()
        self.wordnet = WikiWordnet()

    def get_insight(self, data: df):
        insight = {
            'name': None,
            'company': None,
            'greeted': False,
            'sent_off': False,
            'greeting': None,
            'parting': None,
        }

        for idx, row in data.iterrows():
            raw_text = row['text']
            # Перевод в нижний регистр и избавление от STT-пунктуации
            formatted_text = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
            # Парсинг NLP-моделями - расстановка пунктуации, заглавных и семантический анализ
            processed_text = self.preprocessor(formatted_text, lan='ru')
            doc = self.nlp(processed_text)

            buffer = {}
            # Условие отсечения - отсутствие приветствия в первых 5 репликах считается провалом
            if row['line_n'] < data['line_n'].min() + 5:
                buffer['greeted'], buffer['greeting'] = self.syn_hyp_match(doc, greeting_keywords)

                buffer['name'], buffer['company'] = self.extract_names(doc)

            # Аналогичное отсечение для прощаний
            if row['line_n'] >= data['line_n'].max() - 5:
                buffer['sent_off'], buffer['parting'] = self.syn_hyp_match(doc, parting_keywords)

            for key in buffer:
                if not insight[key]:
                    insight[key] = buffer[key]

        return insight

    def syn_hyp_match(self, doc, keys) -> Tuple[bool, Optional[str]]:
        wordlist = doc.sentences[0].words
        for i in range(1, 3):
            for j in range(0, len(wordlist) - i):
                test = wordlist[j]
                test2 = wordlist[j+i]
                test3 = wordlist[j:j+i]
                phrase = ' '.join([word.text for word in wordlist[j:j+i]]).lower()

                synsets = self.wordnet.get_synsets(phrase)
                synonims = [[word for word in synset.get_words()] for synset in synsets]
                synonims = list(itertools.chain(*synonims))
                syn_lemmas = [word._lemma for word in synonims]

                definitions = [self.nlp(word._definition) for word in synonims]
                def_lemmas = [doc.sentences[0].words for doc in definitions]
                def_lemmas = list(itertools.chain(*def_lemmas))
                def_lemmas = [word._lemma for word in def_lemmas]

                if any(
                        (key in def_lemmas or key in syn_lemmas)
                        and 'алло' not in syn_lemmas
                        for key in keys
                ):
                    return True, doc.text

        return False, None

    def extract_names(self, doc: Document) -> Tuple[str, str]:
        results = {
            'person_name': '',
            'org_name': ''
        }
        # Первая итерация - поиск по результатам NER
        for ent in doc.entities:
            if ent.type == 'PER' and self.check_person(doc, ent.words[0]):
                results['person_name'] = ent.text

            if ent.type == 'ORG':
                results['org_name'] = ent.text

        wordlist = doc.sentences[0].words
        # Альтернативный поиск компании - по ключевому слову и до следующего не-существительного
        if not results['org_name'] and any([word.lemma == 'компания' for word in wordlist]):
            anchor_id = next(word.id for word in wordlist if word.lemma == 'компания')
            org_name = []
            while wordlist[anchor_id].pos in ['NOUN', 'PUNCT']:
                if wordlist[anchor_id].pos == 'NOUN':
                    org_name.append(wordlist[anchor_id].text)
                anchor_id += 1
            results['org_name'] = ' '.join(org_name)

        return results['person_name'], results['org_name']

    def check_person(self, doc: Document, name: Word) -> bool:
        root = self.search_dep(doc, name, 'nsubj', 'VERB')
        pronoun = self.search_dep(doc, root, 'obj', 'PRON') if root else self.search_dep(doc, name, 'nsubj', 'PRON')
        if pronoun and pronoun.lemma in ['я', 'это']:
            return True
        return False

    @staticmethod
    def search_dep(doc: Document, target: Word, target_rel: str, target_pos: str):
        result = None

        valid_deps = [dep for dep in doc.sentences[0].dependencies
                      if dep[0].id == target.id or dep[2].id == target.id]
        relevant_deps = [dep for dep in valid_deps if dep[1] == target_rel]
        for dep in relevant_deps:
            if dep[0].id == target.id and dep[2].pos == target_pos:
                result = dep[2]
            elif dep[2].id == target.id and dep[0].pos == target_pos:
                result = dep[0]

        return result
