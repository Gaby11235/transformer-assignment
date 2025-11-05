import json, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    steps, train, val = [], [], []
    with open(args.log, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            steps.append(rec['step'])
            train.append(rec['train_loss'])
            if 'val_loss' in rec:
                val.append(rec['val_loss'])

    # plot (single figure, no specified styles/colors)
    plt.figure()
    plt.plot(steps, train, label='train_loss')
    if val:
        plt.plot(steps[:len(val)], val, label='val_loss')
    plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == '__main__':
    main()
