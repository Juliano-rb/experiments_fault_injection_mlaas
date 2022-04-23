import argparse
import os
import pandas as pd


def prepare_mlm_data(labels, texts, output_file_path, sep_token):
	
	with open(os.path.join(output_file_path, 'mlm_data.txt'), 'w', encoding='utf-8') as f:
		for label, text in zip(labels, texts):
			print(label, text)
			f.write(' '.join([label, sep_token, text]) + '\n')

def main(args):
	data = pd.read_csv(args.data_path).groupby('sentiment').apply(lambda x: x.sample(int(10)))
	prepare_mlm_data(data['sentiment'].tolist(), data['review'].tolist(), args.output_dir, '[SEP]')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
	parser.add_argument('--data_path', default='./test/res/text/classification.csv', help='Data path')
	parser.add_argument('--output_dir', default='./test/res/text', help='File output directory')

	args = parser.parse_args()

	main(args)
