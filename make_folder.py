import argparse
import os


def create_directories(args):
    directory_path = f'Bilder/{args.Ort}/{args.Bereich}/{args.Reihe}'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i in range(1, int(args.Bäume) + 1):
        tree_directory = os.path.join(directory_path, str(i))
        if not os.path.exists(tree_directory):
            os.makedirs(tree_directory)
            os.makedirs(os.path.join(tree_directory, 'Left'))
            os.makedirs(os.path.join(tree_directory, 'Right'))


def main():
    parser = argparse.ArgumentParser(description="Create directories for tree data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("Bäume", help="Number of Trees")
    parser.add_argument("Reihe", help="Row")
    parser.add_argument("Ort", help="Location")
    parser.add_argument("Bereich", help="Area of Row")
    args = parser.parse_args()

    create_directories(args)

    print("Directories created successfully.")


if __name__ == "__main__":
    main()