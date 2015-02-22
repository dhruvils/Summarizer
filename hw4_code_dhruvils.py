#!python

import os
import operator
import math
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

def get_all_files(dir):
    absoluteFileList = []
    for dirpath, dirs, files in os.walk(dir):
        absoluteFileList += [ dirpath + '/' + filename for filename in files]
    return absoluteFileList

def create_ts_files():
    input_dir = "/home1/c/cis530/hw4/dev_input"
    output_dir = "/home1/d/dhruvils/homework4/topic_words/"
    tool_dir = "/home1/c/cis530/hw4/TopicWords-v2/"
    my_dir = "/home1/d/dhruvils/homework4/"

    config_base_string = "==== Do not change these values ====\n"+\
                         "stopFilePath = stoplist-smart-sys.txt\n"+\
                         "performStemming = N\n"+\
                         "backgroundCorpusFreqCounts = bgCounts-Giga.txt\n"+\
                         "topicWordCutoff = 0.1\n"+\
                         "\n"+\
                         "==== Directory to compute topic words on ====\n\n"
    config_outputdir_string = "==== Output File ====\n\n"

    for i in xrange(0, 4):
        for j in xrange(0, 10):
            f = open("config_" +str(i) +str(j), "w")
            f.write(config_base_string)
            f.write("inputDir = " +input_dir +"/dev_" +str(i) +str(j) +"\n\n")
            f.write(config_outputdir_string)
            f.write("outputFile = " +output_dir +"dhruvils_dev_" +str(i) +str(j) +".ts")
            f.close()

            os.system("(cd " +tool_dir +"; java -Xmx1000m TopicSignatures " +my_dir +"config_" +str(i) +str(j) +")")

def load_topic_words(topic_file, n):
    f = open(topic_file, "r")
    topic_words_dict = dict()

    for line in f:
        line_data = line.split()
        if float(line_data[1]) >= 10.0:
            topic_words_dict[line_data[0]] = float(line_data[1])

    sorted_topic_words = sorted(topic_words_dict.items(), key = operator.itemgetter(1), reverse=True)
    top_n = [tup[0] for tup in sorted_topic_words[:n]]
    remaining_list = [tup[0] for tup in sorted_topic_words[n+1:]]
    return (top_n, remaining_list)

def expand_keywords(keylist, candidatelist, ic, outputfile):
    f = open(outputfile, 'w')
    for key_word in keylist:
        key_word_list = wn.synsets(key_word, pos=wn.NOUN)
        if len(key_word_list) != 0:
            temp_lemma = str(key_word_list[0])
            lemma_name = temp_lemma[temp_lemma.find("(") + 2 : temp_lemma.find(")") - 1]
            
            sim_dict = dict()
            for can_word in candidatelist:
                can_word_list = wn.synsets(can_word, pos=wn.NOUN)
                if len(can_word_list) != 0:
                    a = wn.synset(lemma_name)
                    temp = str(key_word_list[0])
                    can_lemma_name = temp[temp.find("(") + 2 : temp.find(")") - 1]
                    b = wn.synset(can_lemma_name)

                    sim = float(a.res_similarity(b, ic))
                    if sim > 0.0:
                        sim_dict[can_word] = sim

            sorted_sim_list = sorted(sim_dict.items(), key = operator.itemgetter(1), reverse=True)
            word_list = " ".join([can[0] for can in sorted_sim_list])
            f.write(key_word +": " +word_list +"\n")
    f.close()

def summarize_baseline(directory, outputfile):
    file_list = get_all_files(directory)
    outputstring = ""
    for file_name in file_list:
        f = open(file_name)
        if len(outputstring.split()) < 100:
            outputstring += f.readline()
        f.close()

    f = open(outputfile, 'w')
    f.write(outputstring)
    f.close()

def get_unigrams(sentences):
    sentences = sentences.replace(",", "").replace(".", "").replace("(", "").replace(")","").replace("'", "").replace('"', "")
    token_list = word_tokenize(sentences)
    token_list = [word.lower() for word in token_list]

    #word_list = sentences.split()
    #total_words = len(word_list)

    #word_list = sentences.split(" ")
    #total_words = len(word_list)

    f = open("/home1/c/cis530/hw4/stopwords.txt", 'r')
    stopword_string = f.read().split('\n')
    f.close()

    unigram_count = dict()
    total_words = 0
    for word in token_list:
        if word not in stopword_string:
            total_words += 1
            if word not in unigram_count:
                unigram_count[word] = 1
            else:
                unigram_count[word] += 1
    
    for key in unigram_count:
        unigram_count[key] /= float(total_words)
    
    return unigram_count

def calc_kl(total_unigrams, sentence_unigrams):
    kl_val = 0
    for word in sentence_unigrams:
        kl_val += float(sentence_unigrams[word] * math.log(sentence_unigrams[word] / float(total_unigrams[word])))
    return kl_val

def summarize_kl(inputdir, outputfile):
    file_list = get_all_files(inputdir)
    complete_corpus = ""
    for file_name in file_list:
        complete_corpus += open(file_name).read() +"\n"

    total_unigrams = get_unigrams(complete_corpus)

    summary = ""
    summary_sentences = []
    while len(summary.split()) < 100:
        kl_values = []
        for sentence in complete_corpus.split('\n'):
            if not sentence == "":
                if not sentence in summary_sentences:
                    summary_unigram = get_unigrams(summary + "\n" +sentence)
                    if len(summary_unigram) != 0:
                        temp_tuple = (sentence, calc_kl(total_unigrams, summary_unigram))
                        if temp_tuple not in kl_values:
                            kl_values.append(temp_tuple)

        min_kl_sent = min(kl_values, key = operator.itemgetter(1))[0]
        summary += min_kl_sent +"\n"
        summary_sentences.append(min_kl_sent)

    f = open(outputfile, 'w')
    f.write(summary)
    f.close()


def main():
    #for 1.1.1:
    #create_ts_files()

    #for 1.1.2:
    #topic_file = "/home1/d/dhruvils/homework4/topic_words/dhruvils_dev_00.ts"
    #n = 20
    #print load_topic_words(topic_file, n)

    #for 1.2:
    #topic_file = "/home1/d/dhruvils/homework4/topic_words/dhruvils_dev_00.ts"
    #n = 20
    #(keylist, candidatelist) = load_topic_words(topic_file, n)
    #brown_ic = wordnet_ic.ic('ic-brown.dat')
    #outputfile = "./expanded_keywords/dhruvils_dev_00_sample.txt"
    #expand_keywords(keylist, candidatelist, brown_ic, outputfile)

    #for 2.1:
    #for i in xrange(0, 4):
    #    for j in xrange(0, 10):
    #        directory = "/home1/c/cis530/hw4/dev_input/dev_" +str(i) +str(j)
    #        outputfile = "./baseline_summary/sum_dev_" +str(i) +str(j) +".txt"
    #        summarize_baseline(directory, outputfile)

    #for 2.2:
    #for i in xrange(0, 4):
    #    for j in xrange(0, 10):
    #        inputdir = "/home1/c/cis530/hw4/dev_input/dev_" +str(i) +str(j)
    #        outputfile = "./kl/sum_dev_" +str(i) +str(j) +".txt"
    #        summarize_kl(inputdir, outputfile)

if __name__ == "__main__":
    main()
