# transfer TextVQA to VQA format
import json

DATA_DIR = 'dataTVQA'
for split_name in ('train', 'val', 'test'):
    jsdata=json.load(open('{}/TextVQA_0.5_{}.json'.format(DATA_DIR, split_name),'r'))
    new_jsdata={}
    new_jsdata['data_subtype'] = jsdata['dataset_type']
    new_jsdata['data_type'] = 'OpenImages'
    new_jsdata['info'] = {'dataset_name': jsdata['dataset_name'], 'version': jsdata['dataset_version']}

    pairs = jsdata['data']
    print('[Info] {} - #question: {}'.format(jsdata['dataset_type'], len(pairs)))

    # questions
    new_jsdata['questions'] = []
    for pair in pairs:
        new_jsdata['questions'].append({'image_id': pair['image_id'], 'question': pair['question'], 'question_id': pair['question_id']})

    fname = '{}/Questions_TextVQA_{}.json'.format(DATA_DIR, split_name)
    print('[Store] {}'.format(fname))
    json.dump(new_jsdata, open(fname, 'w'))

    # annotations
    if 'answers' in pairs[0]: # train or val set
        del new_jsdata['questions']
        new_jsdata['annotations'] = []
        for pair in pairs:
            new_ans = []
            for idx, ans in enumerate(pair['answers']):
                new_ans.append({'answer': ans, 'answer_id': idx+1})
            new_jsdata['annotations'].append({'image_id': pair['image_id'], 'answers': new_ans, 'question_id': pair['question_id']})

        fname = '{}/Annotations_TextVQA_{}.json'.format(DATA_DIR, split_name)
        print('[Store] {}'.format(fname))
        json.dump(new_jsdata, open(fname, 'w'))
