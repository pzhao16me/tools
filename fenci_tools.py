import helper
import problem_unittests as tests
import jieba

source_path = 'data/test/answer.txt'
target_path = 'data/test/response.txt.'
# source_text = helper.load_data(source_path)
# target_text = helper.load_data(target_path)
def get_lines(filename):
	with open(filename,encoding="utf-8",errors ="ignore") as f:
		lines=[]
		sentences = f.readlines()
		lines=[[ word for word in "".join(jieba.cut(sentence,cut_all = False))] 
				for _,sentence in enumerate(sentences)]
		return lines
    
def get_vocab(filename):
	lines=[]
	with open(filename,encoding="utf-8",errors="ignore") as f:
	 	sentences = f.readlines()
	 	
	 	
	 	for idx ,sentence in enumerate(sentences):
	 		#print(idx,"-->",sentence)
	 		seg_list = jieba.cut(sentence,cut_all = False)
	 		seg_list = "".join(seg_list)
	 		line=[]
	 		for word in seg_list:
	 			line.append(word)
	 		#print(line)
	 		#print(idx,len(line))	
	 		lines.extend(line)
	 	vocab = set(lines)
	 	word_int={}
	 	word_int['<PAD>']=0
	 	word_int['<EOS>']=1
	 	word_int['<UNK>']=2
	 	word_int['<GO>']=3
	 	for idx ,item in enumerate(vocab):
	 		word_int[item]= idx+2
	 	return word_int
    
view_sentence_range = (0, 10)
source_text = get_lines(source_path)
target_text = get_lines(target_path)
vocab_size = len(get_vocab(source_path))
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(vocab_size))

sentences = len(source_text)
word_counts = [len(item) for item in source_text]
print('Number of sentences: {}'.format(sentences))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print("the 10 sentences of source_text")
# show_example(source_text)
# show_example(target_text)
      

def show_example(text):
    print("begin to show some example " ,'\n')
    
    for i in range(10):
        print(text[i])

show_example(source_text)       


def get_word2vec(filename):
    word_int = get_vocab(source_path)
    with open(filename,encoding="utf-8",errors="ignore") as f:
        sentences = f.readlines()
        vocab_to_int =[[word_int.get(word, word_int["<UNK>"])
                        for word in "".join(jieba.cut(sentence,cut_all = False))]
                       for _,sentence in enumerate(sentences)]
        return vocab_to_int



def text_to_ids(source_path, target_path):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_int_text = get_word2vec(source_path)
    target_int_text = get_word2vec(target_path)
    return source_int_text, target_int_text



source_int_text, target_int_text = text_to_ids(source_path, target_path)


# show t 10 sample of the source_int_text
len(source_int_text)
for i in range(10):
    print(source_int_text[i])





# show t 10 sample of the source_int_text
len(source_int_text)
for i in range(10):
    print(source_int_text[i])    

