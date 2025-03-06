#!/usr/bin/env python3
"""
Frequency Analyzer - Processes a sequence of integers and analyzes their distribution
"""

def analyze_sequence():
    """
    Analyzes a sequence of integers and computes various metrics based on their frequency.
    
    The function reads a sequence length n, followed by n integers, and for each possible
    value from 0 to n, it computes and outputs a metric based on frequency and uniqueness.
    """
    # Read the sequence length
    sequence_length = int(input().strip())
    
    # Read the sequence elements
    elements = list(map(int, input().strip().split()))
    
    # Create a dictionary to store element occurrences
    # This is different from the C++ approach which used a fixed-size vector
    occurrence_map = {}
    
    # Count occurrences of each element
    for element in elements:
        if element in occurrence_map:
            occurrence_map[element] += 1
        else:
            occurrence_map[element] = 1
    
    # Track unique elements seen up to each index
    # Using a set-based approach instead of cumulative counting
    unique_values = set()
    unique_counts = [0] * (sequence_length + 1)
    
    for i in range(sequence_length + 1):
        if i in occurrence_map:
            unique_values.add(i)
        unique_counts[i] = len(unique_values)
    
    # Calculate and output results
    for value in range(sequence_length + 1):
        # For each possible value, calculate the result
        
        # Number of missing values before current value
        missing_count = 0
        if value > 0:
            missing_count = value - unique_counts[value - 1]
        
        # Current value's frequency (0 if not present)
        current_frequency = occurrence_map.get(value, 0)
        
        # Take maximum of missing values and current frequency
        result = max(missing_count, current_frequency)
        
        print(result)

if __name__ == "__main__":
    analyze_sequence() 