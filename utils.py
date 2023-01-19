import csv
import string


def calc_steps(df, args):
    steps_per_epoch = len(df) // args.batch_size
    total_training_steps = steps_per_epoch * args.n_epochs
    warmup_steps = total_training_steps // 5
    return total_training_steps, warmup_steps


def get_label_colums(df, args):
    label_columns = df[args.label_column].unique()
    return label_columns


def is_csv(infile):
    try:
        with open(infile, newline='') as csvfile:
            start = csvfile.read(4096)

            # isprintable does not allow newlines, printable does not allow umlauts...
            if not all([c in string.printable or c.isprintable() for c in start]):
                return False
            dialect = csv.Sniffer().sniff(start)
            return True
    except csv.Error:
        # Could not get a csv dialect -> probably not a csv.
        return False

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False