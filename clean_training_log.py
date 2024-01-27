'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-01-26
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-26
 '''
import sys


def remove_lines_with_character(character_to_remove, file_path, new_file_path=None):
    with open(file_path, 'r') as file:    # read the file
        lines = file.readlines()
    filtered_lines = [line for line in lines if character_to_remove not in line]    # filter out lines containing the specified character
    if new_file_path is None: new_file_path = file_path
    with open(new_file_path, 'w') as file:    # write the filtered lines back to the file
        file.writelines(filtered_lines)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python clean_training_log.py <filepath>')
    else:
        filepath = sys.argv[1]
        print(f'Cleaning up: {filepath}')
        remove_lines_with_character('', filepath)
