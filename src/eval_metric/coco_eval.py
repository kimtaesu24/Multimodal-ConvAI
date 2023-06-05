from eval_metric.pycocoevalcap.eval import COCOEvalCap
from eval_metric.pycocotools.coco import COCO


def _coco_evaluation(predicts, answers):
    coco_res = []
    ann = {'images': [], 'info': '', 'type': 'captions', 'annotations': [], 'licenses': ''}

    for i, (predict, answer) in enumerate(zip(predicts, answers)):
        #predict_cap = ' '.join(predict)
        #answer_cap = ' '.join(answer).replace('_UNK', '_UNKNOWN')

        ann['images'].append({'id': i})
        ann['annotations'].append({'caption': answer, 'id': i, 'image_id': i})
        coco_res.append({'caption': predict, 'id': i, 'image_id': i})

    coco = COCO(ann)
    coco_res = coco.loadRes(coco_res)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval.eval


def calculate_eval_matric(output='Test word', ref='Testing word') -> dict:
    # ref = [
    #     ref.replace('!','').replace('.','')
    #     ]
    # output = [ 
    #         output.replace('!','').replace('.','')
    #         ]
    
    eval_results = _coco_evaluation(output, ref)
    # print(eval_results) #### <- 점수 확인
    return eval_results

'''
Return 되는 eval_results 의 결과는 아래 처럼 나옴.
eval_results['Bleu_1'] 이런식으로 가져오면 될 듯함.

{'Bleu_1': 0.1353352829659427, 
'Bleu_2': 0.00013533528303361024, 
'Bleu_3': 1.3533528305616618e-05, 
'Bleu_4': 4.2796774227674215e-06, 
'METEOR': 0.14814814814814814, 
'ROUGE_L': 0.45864661654135336, 
'CIDEr': 0.0, 'SPICE': 0.0}
'''

if __name__ == '__main__':
    calculate_eval_matric()
