def getIntersection(list1, list2):
    list1.sort()
    list2.sort()

    intersections = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        start1, end1 = list1[i]
        start2, end2 = list2[j]

        start_intersection = max(start1, start2)
        end_intersection = min(end1, end2)

        if start_intersection <= end_intersection:
            intersections.append((start_intersection, end_intersection))

        if end1 < end2:
            i += 1
        else:
            j += 1

    return intersections

def merge_ranges(ranges):
    if not ranges:
        return []
    
    ranges.sort()
    merged = [ranges[0]]

    for current in ranges[1:]:
        last_merged = merged[-1]
        
        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged

def getUnion(list1, list2):
    combined = list1 + list2
    return merge_ranges(combined)

def getComplement(intervals):
    result = []
    current_start = float('-inf')
    
    for interval in sorted(intervals):
        if current_start < interval[0]:
            result.append((current_start, interval[0]))
        current_start = max(current_start, interval[1])
    
    if current_start < float('inf'):
        result.append((current_start, float('inf')))
    
    return result
