class WordBuilder:

    def __init__(self):
        self.word = ""

    # ==========================================
    # Add a recognized gesture
    # ==========================================

    def add(self, prediction):

        if prediction is None:
            return self.word

        prediction = str(prediction)

        # Only add single-character gestures
        if len(prediction) == 1:
            self.word += prediction

        return self.word

    # ==========================================
    # Add space
    # ==========================================

    def space(self):

        if self.word and not self.word.endswith(" "):
            self.word += " "

        return self.word

    # ==========================================
    # Remove last character
    # ==========================================

    def backspace(self):

        if self.word:
            self.word = self.word[:-1]

        return self.word

    # ==========================================
    # Clear entire word
    # ==========================================

    def clear(self):

        self.word = ""

        return self.word

    # ==========================================
    # Get current text
    # ==========================================

    def get_text(self):

        return self.word

    # ==========================================
    # Reset
    # ==========================================

    def reset(self):

        self.word = ""