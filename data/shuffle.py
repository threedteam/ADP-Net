import random
import os


def shuffle_files_synchronously(file1_path, file2_path, output1_path, output2_path, seed=None):
    lines1 = []
    lines2 = []
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2_path, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise

    if len(lines1) != len(lines2):
        raise ValueError(f"Error: File '{file1_path}' ({len(lines1)} lines) and "
                         f"'{file2_path}' ({len(lines2)} lines) have different number of lines, cannot shuffle synchronously.")

    print(f"Successfully read {len(lines1)} lines.")

    paired_lines = list(zip(lines1, lines2))

    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")

    print("Shuffling lines...")
    random.shuffle(paired_lines)
    print("Shuffling complete.")

    shuffled_lines1, shuffled_lines2 = zip(*paired_lines)

    shuffled_lines1 = list(shuffled_lines1)
    shuffled_lines2 = list(shuffled_lines2)

    print(f"Writing shuffled lines to '{output1_path}' and '{output2_path}'...")
    try:
        with open(output1_path, 'w', encoding='utf-8') as out1:
            out1.writelines(shuffled_lines1)
        with open(output2_path, 'w', encoding='utf-8') as out2:
            out2.writelines(shuffled_lines2)
        print("Writing complete.")
    except Exception as e:
        print(f"An error occurred while writing to files: {e}")
        raise


if __name__ == "__main__":

    input_file1 = "train_img.txt"
    input_file2 = "train_instance_mask.txt"
    output_file1 = "train_img.txt"
    output_file2 = "train_instance_mask.txt"

    print("\n--- Starting synchronous shuffle ---")
    try:
        shuffle_files_synchronously(input_file1, input_file2, output_file1, output_file2, seed=3407)
        print("\nSynchronous shuffle successful!")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nShuffle failed: {e}")
    except Exception as e:
        print(f"\nAn unknown error occurred: {e}")