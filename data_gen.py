import itertools
from fractions import Fraction
import csv
import random
import os

def can_make_24(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) < 1e-6
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j:
                a, b = nums[i], nums[j]
                rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
                next_nums_list = [
                    rest + [a + b],
                    rest + [a - b],
                    rest + [b - a],
                    rest + [a * b]
                ]
                if b != 0:
                    next_nums_list.append(rest + [Fraction(a, b)])
                if a != 0:
                    next_nums_list.append(rest + [Fraction(b, a)])
                
                for next_nums in next_nums_list:
                    if can_make_24(next_nums):
                        return True
    return False

def generate_data():
    valid_combinations = []
    # Generate all combinations with replacement of 4 numbers from 1 to 13
    for comb in itertools.combinations_with_replacement(range(1, 14), 4):
        if can_make_24([Fraction(x) for x in comb]):
            valid_combinations.append(comb)
            
    print(f"Found {len(valid_combinations)} valid 24-point combinations.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_combinations)
    
    split_idx = int(len(valid_combinations) * 0.9)
    train_data = valid_combinations[:split_idx]
    test_data = valid_combinations[split_idx:]
    
    os.makedirs('data', exist_ok=True)
    
    with open('data/train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['nums'])
        for comb in train_data:
            writer.writerow([", ".join(map(str, comb))])
            
    with open('data/test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['nums'])
        for comb in test_data:
            writer.writerow([", ".join(map(str, comb))])
            
    print("Data saved to data/train.csv and data/test.csv")

if __name__ == "__main__":
    generate_data()
