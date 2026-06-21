"""
Character vocabularies and index mappings for Javanese and Sundanese CRNN OCR models.
"""

from __future__ import annotations

# Javanese Unicode Script Map
JAWA_CHARS: dict[str, str] = {
    # Consonants
    "\uA9B2": "ha", "\uA9A4": "na", "\uA995": "ca", "\uA9AB": "ra", "\uA98F": "ka",
    "\uA9A2": "da", "\uA9A0": "ta", "\uA9B1": "sa", "\uA9AE": "wa", "\uA9AD": "la",
    "\uA9A5": "pa", "\uA99D": "dha", "\uA997": "ja", "\uA9AA": "ya", "\uA99A": "nya",
    "\uA9A9": "ma", "\uA992": "ga", "\uA9A7": "ba", "\uA99B": "tha", "\uA994": "nga",
    # Digits
    "\uA9D0": "angka_0", "\uA9D1": "angka_1", "\uA9D2": "angka_2", "\uA9D3": "angka_3",
    "\uA9D4": "angka_4", "\uA9D5": "angka_5", "\uA9D6": "angka_6", "\uA9D7": "angka_7",
    "\uA9D8": "angka_8", "\uA9D9": "angka_9",
    # Sandhangan (Vowels/Diacritics)
    "\uA9B6": "wulu",      # i (e.g. gi, ti, ri)
    "\uA9B8": "suku",      # u (e.g. tu, ru)
    "\uA9BA": "taling",    # é
    "\uA9BC": "pepet",     # ê
    "\uA9C0": "pangkon",   # killer (ends word consonant)
    "\uA981": "cecak",     # ng (final consonant)
    "\uA982": "layar",     # r (final consonant)
    "\uA983": "wignyan",   # h (final consonant)
}

# Sundanese Unicode Script Map
SUNDA_CHARS: dict[str, str] = {
    # Swara (Vowels)
    "\u1B83": "swara_a", "\u1B84": "swara_i", "\u1B85": "swara_u", "\u1B86": "swara_e_accent",
    "\u1B87": "swara_o", "\u1B88": "swara_e", "\u1B89": "swara_eu",
    # Ngalagena (Consonants)
    "\u1B8A": "nga_ka", "\u1B8B": "nga_qa", "\u1B8C": "nga_ga", "\u1B8D": "nga_nga",
    "\u1B8E": "nga_ca", "\u1B8F": "nga_ja", "\u1B90": "nga_za", "\u1B91": "nga_nya",
    "\u1B92": "nga_ta", "\u1B93": "nga_da", "\u1B94": "nga_na", "\u1B95": "nga_pa",
    "\u1B96": "nga_fa", "\u1B97": "nga_va", "\u1B98": "nga_ba", "\u1B99": "nga_ma",
    "\u1B9A": "nga_ya", "\u1B9B": "nga_ra", "\u1B9C": "nga_la", "\u1B9D": "nga_wa",
    "\u1B9E": "nga_sa", "\u1B9F": "nga_xa", "\u1BA0": "nga_ha",
    "\u1BAE": "nga_kha", "\u1BAF": "nga_sya",
    # Digits
    "\u1BB0": "digit_0", "\u1BB1": "digit_1", "\u1BB2": "digit_2", "\u1BB3": "digit_3",
    "\u1BB4": "digit_4", "\u1BB5": "digit_5", "\u1BB6": "digit_6", "\u1BB7": "digit_7",
    "\u1BB8": "digit_8", "\u1BB9": "digit_9",
    # Rarangken (Vowel/Consonant signs)
    "\u1BA4": "rarangken_panghulu",    # i
    "\u1BA5": "rarangken_panyuku",     # u
    "\u1BA6": "rarangken_paneleng",    # é
    "\u1BA7": "rarangken_panolong",    # o
    "\u1BA8": "rarangken_pamepet",     # e
    "\u1BA9": "rarangken_paneuleung",  # eu
    "\u1B80": "rarangken_panyecek",    # ng
    "\u1B81": "rarangken_panglayar",   # r
    "\u1B82": "rarangken_pangwisad",   # h
    "\u1BA1": "rarangken_pamingkal",   # y
    "\u1BA2": "rarangken_panyakra",    # r
    "\u1BA3": "rarangken_panyiku",     # l
    "\u1BAC": "rarangken_pamintel",    # m
    "\u1BAD": "rarangken_papasangan",  # w
    "\u1BAA": "rarangken_paten",       # killer
    "\u1BBA": "avagraha",
}

class ScriptVocab:
    def __init__(self, script_type: str):
        self.script_type = script_type.lower()
        if self.script_type == "jawa":
            self.char_to_label = JAWA_CHARS
        elif self.script_type == "sunda":
            self.char_to_label = SUNDA_CHARS
        else:
            raise ValueError(f"Unknown script: {script_type}")
        
        self.label_to_char = {v: k for k, v in self.char_to_label.items()}
        
        # Sort labels alphabetically for deterministic indices
        self.labels = sorted(self.char_to_label.values())
        
        # Mappings (0 to N-1)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
        
        self.num_classes = len(self.labels)
        
    def encode_string(self, text_list: list[str]) -> list[int]:
        """Convert a list of labels to a list of integer indices."""
        return [self.label_to_idx[lbl] for lbl in text_list]
        
    def decode_indices(self, indices: list[int]) -> list[str]:
        """Convert a list of integer indices to a list of labels."""
        return [self.idx_to_label[idx] for idx in indices if idx in self.idx_to_label]

    def decode_to_unicode(self, indices: list[int]) -> str:
        """Convert a list of integer indices to a combined Unicode string."""
        chars = []
        for idx in indices:
            if idx in self.idx_to_label:
                lbl = self.idx_to_label[idx]
                chars.append(self.label_to_char.get(lbl, lbl))
        return "".join(chars)
