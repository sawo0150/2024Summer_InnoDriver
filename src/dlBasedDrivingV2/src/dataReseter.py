#!/usr/bin/env python3

import os
import csv

def clear_labels(csv_path):
    rows = []
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Clear the values of columns -2, -1, 0, 1, 2
            row['-2'] = ''
            row['-1'] = ''
            row['0'] = ''
            row['1'] = ''
            row['2'] = ''
            rows.append(row)

    with open(csv_path, mode='w', newline='') as file:
        fieldnames = ['image', '-2', '-1', '0', '1', '2']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def process_folders(base_dir):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    print("Available folders:")
    for i, folder in enumerate(folders):
        print(f"{i}: {folder}")
    
    index = int(input("Select the folder index: "))
    selected_folder = folders[index]
    
    csv_dir = os.path.join(base_dir, selected_folder, 'csv2')
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        clear_labels(csv_path)
        print(f"Cleared labels in {csv_file}")

def main():
    base_dir = '/home/innodriver/InnoDriver_ws/Data'
    process_folders(base_dir)

if __name__ == "__main__":
    main()
