import argparse
from matrix import cycle_graph

from train import *


def main():
    agents = 5
    w = cycle_graph(agents)

    # dataset="cifar10"
    dataset = "imagenet"
    epochs = 20
    bs = 128
    seed = 42

    def parse_args():
        """ Function parses command line arguments """
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--test_num", default=0, type=int)
        parser.add_argument("-r", "--run_num", default=0, type=int)
        parser.add_argument("-s", "--stratified", default=True)
        return parser.parse_args()

    args = parse_args()
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    stratified = args.stratified
    fname = os.path.join(results_path, f"{dataset}_e{epochs}_hom{stratified}_{args.test_num}.csv")

    print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.test_num == 0:
        Algorithm2Trainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                          agents=agents)
    elif args.test_num == 1:
        DSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                     agents=agents)
    elif args.test_num == 2:
        DADAMTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                     agents=agents)
    elif args.test_num == 3:
        DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                        agents=agents)
    elif args.test_num == 4:
        ATCDIGingTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
    elif args.test_num == 5:
        DSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)


if __name__ == "__main__":
    main()
