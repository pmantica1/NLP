from tqdm import tqdm
import csv

def get_all_words():
    words = set()
    count = 0
    with open("glove.42B.300d.txt") as infile:
        for line in tqdm(infile):
            count += 1
            parsed_line = line.split()
            word = parsed_line[0]
            words.add(word)
    return words

def get_seen_words(filename):
    seen_words = set()
    count = 0
    with open(filename) as infile:
        reader = csv.reader(infile, delimiter="\t")
        for row in tqdm(reader):
            count +=1
            title = row[1].lower()
            body = row[2].lower()
            for word in title.split():
                seen_words.add(word)
            for word in body.split():
                seen_words.add(word)
    return seen_words


def get_important_words():
    all_words = get_all_words()
    seen_words = get_seen_words("../data/texts_raw_fixed.txt").union(get_seen_words("../android_data/corpus.txt"))
    return all_words.intersection(seen_words)


def prune(important_words):
    count =0
    with open("glove.42B.300d.txt") as infile:
        with open("glove_prunned.txt", "w") as outfile:
            for line in tqdm(infile):
                count += 1
                parsed_line = line.split()
                word = parsed_line[0]
                if word in important_words:
                    outfile.write(line)
if __name__ == "__main__":
    prune(get_important_words())