import re
import os
import sys
import csv


def save_as_csv(scores_save, file_name):
	with open(file_name, 'wb') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=scores_save.keys())

		writer.writeheader()
		for i in range(0, len(scores_save['METEOR'])):
			row_dict = {}
			for (metric, scores) in scores_save.iteritems():
				row_dict[metric] = scores[i]
			writer.writerow(row_dict)

def main():
	file_to_parse = sys.argv[1]
	save_name = sys.argv[2]
	scores_save = {
		'Bleu_1': [],
		'Bleu_2': [],
		'Bleu_3': [],
		'Bleu_4': [],
		'METEOR': [],
		'ROUGE_L': [],
		'CIDEr': []
	}
	key_sequencs = ['Bleu_1',
		'Bleu_2',
		'Bleu_3',
		'Bleu_4',
		'METEOR',
		'ROUGE_L',
		'CIDEr']
	with open(file_to_parse, 'rb') as f:
		lines = f.readlines()
		i = 0
		while (i < len(lines)):
			if (lines[i].find('Epoch') != -1):
				for key in key_sequencs:
					print "searc line: ", lines[i+1]
					print "regex template:", '{}: (.*)'.format(key)
					search_obj = re.search( r'{}: (.*)'.format(key), lines[i+1])
					scores_save[key].append(float(search_obj.group(1)))
					i += 1
			else:
				i += 1

	save_as_csv(scores_save, save_name)
	return

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: python {} file_to_parse save_name".format(sys.argv[0])
		exit()
	main()
