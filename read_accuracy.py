import os
import json
import argparse
import xlsxwriter
from xlsxwriter.utility import xl_range, xl_rowcol_to_cell


parser = argparse.ArgumentParser()
parser.add_argument('cp_dir', help='checkpoint directory where the accuracy files are stored')


def main():
    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    all_files = os.listdir(args.cp_dir)
    acc_files = filter(lambda fn: fn.startswith('accuracy'), all_files)

    def get_iter(af):
        return int(af.split('-', 1)[1].split('.', 1)[0])
    acc_files = sorted(acc_files, key=get_iter)

    iteration = []
    overall = []
    perAnswerType = {}
    perQuestionType = {}
    for af in acc_files:
        iteration.append(get_iter(af))
        with open(os.path.join(args.cp_dir, af)) as f:
            data = json.load(f)

        overall.append(data['overall'])

        for k, v in data['perAnswerType'].items():
            if k not in perAnswerType:
                perAnswerType[k] = []
            perAnswerType.get(k, []).append(v)

        for k, v in data['perQuestionType'].items():
            if k not in perQuestionType:
                perQuestionType[k] = []
            perQuestionType.get(k, []).append(v)

    print('iter\tacc')
    for i, a in zip(iteration, overall):
        print('{}\t{}'.format(i, a))

    N = len(iteration)
    workbook = xlsxwriter.Workbook('accuracy_summary.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'checkpoint:')
    worksheet.write(0, 1, os.path.normpath(args.cp_dir).split('/')[-1])

    def write_line(row, head, acc_list):
        worksheet.write(row, 0, head)
        if len(acc_list) > 0:
            max_cell = xl_rowcol_to_cell(row, 1)
            max_range = xl_range(row, 2, row, N+1)
            worksheet.write_formula(max_cell, '=MAX(%s)' % (max_range,))
        for i, acc in enumerate(acc_list):
            worksheet.write(row, i+2, acc)

    write_line(1, 'Iteration', iteration)
    write_line(2, 'Overall', overall)
    write_line(3, 'AnswerType', [])
    write_line(4, 'yes/no', perAnswerType['yes/no'])
    write_line(5, 'number', perAnswerType['number'])
    write_line(6, 'other', perAnswerType['other'])
    write_line(7, 'QuestionType', [])
    row = 8
    perQuestionType = sorted([(k,v) for k,v in perQuestionType.items()])
    for k, v in perQuestionType:
        write_line(row, k, v)
        row += 1

    worksheet.write(1, 1, 'Max')

    # color it
    for i in range(2, row):
        worksheet.conditional_format(i, 1, i, N+1, {'type': '3_color_scale',
                                                  'min_value': 0.0,
                                                  'mid_value': 50.0,
                                                  'max_value': 100.0})

    # formatting
    bold = workbook.add_format({'bold': True})
    worksheet.set_column(0, 0, 30, bold)
    worksheet.set_row(0, None, bold)
    worksheet.set_row(1, None, bold)

    workbook.close()


if __name__ == '__main__':
    main()

