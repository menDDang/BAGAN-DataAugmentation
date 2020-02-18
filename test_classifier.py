import argparse
from utils.data_loader import *
from utils.hparams import HParam
from models.classifier import *
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--chkpt_dir', required=True, type=str)

    args = parser.parse_args()
    hp = HParam(args.config)

    # Create data sets
    test_dataset = Dataset(mode='testing', hp=hp)

    # Build end-to-end classifier
    model = ResNetClassifier(hp)

    # Load model
    chkpt = args.chkpt_dir
    model.load_weights(chkpt).expect_partial()

    y_pred = dict()
    for key, x in test_dataset.x.items():
        tmp = []
        i = 0
        while True:
            if i + 100 < len(x):
                tmp += [np.array(model(x[i:i+100]), dtype=np.float32)]
                i += 100
            else:
                tmp += [np.array(model(x[i:]), dtype=np.float32)]
                break

        tmp = np.vstack(tmp)
        y_pred[key] = tmp

    mean_eer = []
    for key, x in test_dataset.x.items():
        if key == 'unknown':
            continue

        y_true = commands[key]
        eer, diff = 1, 1
        for threshold in [i*0.01 for i in range(100)]:
            correct_num, error_num = 0, 0
            for j in range(len(y_pred[key])):
                if y_pred[key][j, y_true] > threshold:
                    correct_num += 1
                else:
                    error_num += 1
            FAR = float(error_num) / len(y_pred[key])

            correct_num, error_num = 0, 0
            for j in range(len(y_pred['unknown'])):
                if y_pred['unknown'][j, y_true] > threshold:
                    error_num += 1
                else:
                    correct_num += 1
            FFR = float(error_num) / len(y_pred['unknown'])

            if diff > abs(FAR - FFR):
                diff = abs(FAR - FFR)
                eer = (FAR+FFR)/2

        #print(key, eer)
        mean_eer.append(eer)

    mean_eer = np.mean(mean_eer)
    print(mean_eer)


