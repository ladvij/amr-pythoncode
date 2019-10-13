"""
This program uses Amazon EMR and S3 services to calculate the conditional 
probabilty that a word w' occurs immediately after another word w for each 
and every two-word sequence in the entire collection of over 200K short jokes
from Kaggle:
https://www.kaggle.com/abhinavmoudgil195/short-jokes.

It ultimately returns the top ten probabilities for a target word.

A test file jokestest2-.csv is used for testing.

Usage:
To run it locally with the testfile:
python strips-bigram.py jokestest20.csv

To run it on AWS using EMR and S3 buckets for source and output:
python stripes-bigram.py -r emr s3://<bucketname>/<folder>/jokestest20.csv \
    --output-dir=s3://<bucketname>/stripes.out
    --no-output
    -c MRJob.conf

where MRJob.conf contains the AWS access keys and specifies the EMR instance type and region
"""

__author__ = "Vijay Lad"

from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import Counter
import string
import operator

TARGET_WORD = "me"
DUMCHAR = "*"  # used to represent count of the target word

translator = str.maketrans('','',string.punctuation) # use to remove punctuation

class MRStripes(MRJob):
    """
    Partitioner should send all identical keys to the same reducer
    """

    def mapper(self, _, line):
        """
        E.g. for line:
        [me narrating a documentary about narrators] "I can't hear me what they're saying cuz I'm talking"

        Yields key, values:
        KEY             VALUE
        "me"            {"*":1 , "narrating": 1, "what": 1} # "narrating" and "what" follow "me"
        "narrating"     {"*": 1, "a": 1}
        "a"             {"*": 1, "documentary": 1}
        "documentary"   {"*": 1, "about": 1}
        "about"         {"*": 1, "narrators": 1}
        narrators       {"*": 1, "i": 1}
        i               {"*": 1, "cant": 1}
        cant            {"*": 1, "hear": 1}
        hear            {"*": 1, "me": 1}
        me              {"*": 1}           # 2nd occurence of "me"
        "what"          {"*": 1, "theyre": 1}
        "saying"        {"*": 1, "cuz": 1}
        "cuz"           {"*": 1, "im": 1}
        "im"            {"*": 1, "talking": 1}
        "talking"       {"*": 1}           # last word in sentance
        """
        sublines = line.split(",",1) # remove the first column
        jokeline = sublines[1].translate(translator)
        line = jokeline.strip().lower().split()

        line_length = len(line)

        processed = {}
        t_stripes = {}
        for word in range(line_length):
            if line[word] not in processed:
                processed[line[word]] = 1
                t_stripes = {DUMCHAR:1}
                for neighbour in range(word+1, len(line)):
                    if line[neighbour-1] == line[word]:
                        if line[neighbour] in t_stripes:
                            t_stripes[line[neighbour]] += 1
                        else:
                            t_stripes[line[neighbour]] = 1
                yield line[word], t_stripes

            else:
                yield line[word] , {DUMCHAR:1}

        processed.clear()

    def reducer(self,word,strips):
        """
        Processes:
        Word   Strips
        "me" , ( {"*":1,"narrating":1,"what":1},{"*":1},{"*",1},{"*":1,"was":1},{"*":1})

        Calculates the probabilities for all
        Yields key, values like:
        KEY      VALUE
        "a"      {"big":0.5, "small": 0.5}
        "and"    {"a": 0.5, "my":0.5}
        """

        result = Counter()

        for strip in strips:
            for key, value in strip.items():
                result[key] += value # get: "me" {"*":4, "narrating":1, "what":1, "was":1}

            total_w = result[DUMCHAR]  # which is 4 for "me"
        
        for key,value in result.items():
            result[key] = result[key]/total_w

        del result[DUMCHAR] # remove the count of a word i.e. {"*":}


        yield word, result

    def reducer_topten(self,word,probs):
        """
        Extracts the top ten probabilities for the target word
        """
        if word == TARGET_WORD:
            for probdict in probs:
                sorted_dict = sorted(probdict.items(),key=operator.itemgetter(1), reverse=True)
                max_terms = len(sorted_dict)
            if max_terms > 10:
                max_terms = 10
            for rank in range(max_terms):
                yield rank+1, sorted_dict[rank]


    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer),
            MRStep(reducer=self.reducer_topten)
        ]

if __name__=='__main__':
    MRStripes.run()





