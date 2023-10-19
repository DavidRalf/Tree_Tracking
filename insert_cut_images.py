import argparse
import os
import shutil


def move_files(args):
    max_trees = len(os.listdir(args.ordner))
    print(f"Total files to move: {max_trees}")

    for filename in os.listdir(args.ordner):
        name2 = filename[:19]
        count = filename.split("-")[-1][4:-4]
        count = int(count)

        if args.reverse:
            count = max_trees - count + 1

        destination_path = f'Bilder/{args.Ort}/{args.Bereich}/{args.Reihe}/{count}/{args.Seite}/{name2}.png'
        shutil.move(os.path.join(args.ordner, filename), destination_path)
        print(f"Moved: {filename} to {destination_path}")


def main():
    parser = argparse.ArgumentParser(description="Move files to destination folders",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ordner", help="Source folder")
    parser.add_argument("reverse", help="Reverse order", nargs='?', default="False")
    parser.add_argument("Reihe", help="Destination folder")
    parser.add_argument("Seite", help="Left or Right side of the Tree")
    parser.add_argument("Ort", help="Location")
    parser.add_argument("Bereich", help="Area of Row")
    args = parser.parse_args()

    args.reverse = args.reverse == "True"  # Convert to a boolean

    move_files(args)


if __name__ == "__main__":
    main()
