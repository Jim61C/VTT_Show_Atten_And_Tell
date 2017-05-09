import cPickle as pickle 
import os
import sys
import copy
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

FILTER_BLEU4_SCORE = 10**-5
FILTER_METEOR_SCORE = 0.25

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"])
        # (Meteor(),"METEOR"),
        # (Rouge(),"ROUGE_L"),
        # (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def main():
	this_split = sys.argv[1]
	reference_file_template = './data_MSRVTT/{}/{}.references.pkl'

	references = pickle.load(open(reference_file_template.format(this_split, this_split), 'rb'))

	filtered_references = {}
	for (i, annotations) in references.iteritems():
		filtered_annotations = []

		for gt in annotations:
			hypo = {}	
			hypo[0] = [gt]

			annotations_minus_gt = copy.deepcopy(annotations)
			annotations_minus_gt.remove(gt)
			ref = {}
			ref[0] = annotations_minus_gt

			# leave one out score
			final_scores = score(ref, hypo)
			print "gt: ", gt
			print 'Bleu_1:\t',final_scores['Bleu_1']  
			print 'Bleu_2:\t',final_scores['Bleu_2']  
			print 'Bleu_3:\t',final_scores['Bleu_3']  
			print 'Bleu_4:\t',final_scores['Bleu_4']  
			# print 'METEOR:\t',final_scores['METEOR']  
			# print 'ROUGE_L:',final_scores['ROUGE_L']  
			# print 'CIDEr:\t',final_scores['CIDEr']

			if (not (final_scores['Bleu_4'] < FILTER_BLEU4_SCORE)):
			# if (not (final_scores['METEOR'] < FILTER_METEOR_SCORE)):
				filtered_annotations.append(gt)

		print "references[{}]: ".format(i), "number of inconsistent gt:", len(annotations) - len(filtered_annotations), "\n\n"

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python {} split".format(sys.argv[0])
		print "split -- E.g, test"
		exit()
	main()
