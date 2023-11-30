import random
from grossary import words_phonetics

class RandomSortChooser:
    def __init__(self, items):
        self.items = items
        self.shuffle_items()

    def shuffle_items(self):
        self.shuffled_items = self.items.copy()
        random.shuffle(self.shuffled_items)
        # print(self.shuffled_items)
        print([item["word"] for item in self.shuffled_items])
        self.index = 0

    def choose(self):
        if self.index >= len(self.shuffled_items):
            self.shuffle_items()

        chosen_item = self.shuffled_items[self.index]
        self.index += 1
        return chosen_item

chooser = RandomSortChooser(words_phonetics)

