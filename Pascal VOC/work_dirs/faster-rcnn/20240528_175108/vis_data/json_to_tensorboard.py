import json
from tensorboardX import SummaryWriter

TRAIN_METRIC = [
    'loss', 'loss_rpn_cls', 'loss_rpn_bbox',
    'loss_cls', 'loss_bbox'
]

def convert_json_tb(json_path: str='scalars.json', tensorboard_name: str='pascal_voc'):
    writer = SummaryWriter(tensorboard_name)
    
    with open(json_path, 'r') as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            # write validation scores
            try:
                mAP = line[f'{tensorboard_name}/mAP']
                epoch = line['step']
                AP50 = line[f'{tensorboard_name}/AP50']
                writer.add_scalars('Validation', {
                            'mAP': mAP,
                            'AP50': AP50,
                        }, epoch)
            # write training loss and accuracy
            except:
                writer.add_scalars('Training Loss', {
                    metric: line[metric] for metric in TRAIN_METRIC
                }, line['iter'])
                writer.add_scalars('Training Accuracy', {
                    'Accuracy': line['acc']
                }, line['iter'])
    writer.close()
    f.close()
                
if __name__ == '__main__':
    convert_json_tb()
            
