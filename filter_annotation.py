import cPickle as pickle
import os
import sys
import copy
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

FILTER_BLEU4_SCORE = 10**-9
FILTER_METEOR_SCORE = 0.20
FILTER_CIDER_SCOER = 0.4
FILTER_SUM_SCORE = 1.0

def score(ref, hypo, metric = "METEOR"):
    scorers = [
        # (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"])
        # (Meteor(),"METEOR")
        # (Rouge(),"ROUGE_L"),
        # (Cider(),"CIDEr")
    ]
    if (metric == "METEOR"):
    	scorers.append((Meteor(),"METEOR"))
    elif (metric == "ROUGE_L"):
    	scorers.append((Rouge(),"ROUGE_L"))
    elif (metric == "CIDEr"):
    	scorers.append((Cider(),"CIDEr"))
    elif (metric == "Bleu_1" or metric == "Bleu_2" or metric == "Bleu_3" or metric == "Bleu_4"):
    	scorers.append((Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]))
    
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
	this_dataset = sys.argv[1]
	this_metric = sys.argv[2]
	this_split = sys.argv[3]
	reference_file_template = './{}/{}/{}.references.pkl.orig'

	references = pickle.load(open(reference_file_template.format(this_dataset, this_split, this_split), 'rb'))

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
			final_scores = score(ref, hypo, this_metric)
			print "gt: ", gt
			for (key, val) in final_scores.iteritems():
				print key, ": ", val
			# print 'Bleu_1:\t',final_scores['Bleu_1']
			# print 'Bleu_2:\t',final_scores['Bleu_2']
			# print 'Bleu_3:\t',final_scores['Bleu_3']
			# print 'Bleu_4:\t',final_scores['Bleu_4']
			# print 'METEOR:\t',final_scores['METEOR']
			# print 'ROUGE_L:',final_scores['ROUGE_L']
			# print 'CIDEr:\t',final_scores['CIDEr']

			if (this_metric == "Bleu_4"):
				if (not (final_scores['Bleu_4'] < FILTER_BLEU4_SCORE)):
					filtered_annotations.append(gt)	
			elif (this_metric == "METEOR"):
				if (not (final_scores['METEOR'] < FILTER_METEOR_SCORE)):
					filtered_annotations.append(gt)
			elif (this_metric == "CIDEr"):
				if (not (final_scores['CIDEr'] < FILTER_CIDER_SCOER)):
					filtered_annotations.append(gt)
			else:
				# use sum score
				score_sum = final_scores['Bleu_4'] + final_scores['CIDEr'] + final_scores['METEOR']
				if (not (score_sum < FILTER_SUM_SCORE)):
					filtered_annotations.append(gt)

		filtered_references[i] = filtered_annotations

		print "references[{}]: ".format(i), "number of inconsistent gt:", len(annotations) - len(filtered_annotations), "/", len(annotations), "\n\n"

		if (len(filtered_annotations) == 0):
			raise ValueError('Clean Too Much, score has become zero')

	# dump references
	f = open('./{}/{}/{}.references.filtered.{}.pkl'.format(this_dataset, this_split, this_split, this_metric), 'wb')
	pickle.dump(filtered_references, f, protocol = pickle.HIGHEST_PROTOCOL)
	f.close()

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print "Usage: python {} dataset metric split".format(sys.argv[0])
		print "dataset -- E.g. data_MSRVTT"
		print "metric -- E.g., METEOR"
		print "split -- E.g, test"
		exit()
	main()
