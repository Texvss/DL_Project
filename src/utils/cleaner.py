import os
import glob
import argparse

def load_protocol_ids(protocol_path: str) -> set[str]:
    ids = set()
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def clean_split(processed_dir: str, protocol_path: str) -> None:
    proc_files = glob.glob(os.path.join(processed_dir, '*.npy'))
    proc_ids = {os.path.basename(p).rsplit('.', 1)[0] for p in proc_files}
    proto_ids = load_protocol_ids(protocol_path)

    extras = proc_ids - proto_ids
    if not extras:
        print(f"No extras in {processed_dir}")
        return

    for utt in extras:
        file_path = os.path.join(processed_dir, f"{utt}.npy")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed extra: {file_path}")
    print(f"Cleaned {len(extras)} files from {processed_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove .npy files not in protocol')
    parser.add_argument('--mode', choices=['train', 'dev', 'eval'], required=True)
    parser.add_argument('--processed_dir', required=False)
    parser.add_argument('--protocol', required=False)
    parser.add_argument('--data_root', default='data', help='root folder for data')
    args = parser.parse_args()

    protocol_map = {
        'train': 'raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'dev':   'raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'eval':  'raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    }
    proc_map = {
        'train': 'processed/train',
        'dev':   'processed/dev',
        'eval':  'processed/eval'
    }

    proc_dir = args.processed_dir or os.path.join(args.data_root, proc_map[args.mode])
    prot_path = args.protocol or os.path.join(args.data_root, protocol_map[args.mode])

    clean_split(proc_dir, prot_path)
