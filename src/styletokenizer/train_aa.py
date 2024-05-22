import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextClassifier model.')
    args = parser.parse_args()
    main(train_path=args.train_path, dev_path=args.dev_path)
