import json
from collections import defaultdict

# Load COCO annotations
with open('StudyCase/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Create category mapping
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Analyze annotations for small objects
small_objects = []
area_threshold = 400  # 20x20 pixels
size_stats = defaultdict(list)

for ann in coco_data['annotations']:
    area = ann['area']
    bbox = ann['bbox']  # [x, y, width, height]
    width = bbox[2]
    height = bbox[3]
    cat_id = ann['category_id']
    cat_name = categories[cat_id]

    # Store all areas for statistics
    size_stats[cat_name].append(area)

    # Check if smaller than 20x20
    if area < area_threshold:
        small_objects.append({
            'id': ann['id'],
            'category': cat_name,
            'area': area,
            'width': width,
            'height': height,
            'dimensions': f"{width:.1f}x{height:.1f}"
        })

# Sort by area
small_objects.sort(key=lambda x: x['area'])

print("="*80)
print("OBJECTS SMALLER THAN 20x20 PIXELS (area < 400)")
print("="*80)
print(f"\nTotal annotations: {len(coco_data['annotations'])}")
print(f"Small objects (<20x20): {len(small_objects)}")
print(f"Percentage: {100*len(small_objects)/len(coco_data['annotations']):.2f}%")

if small_objects:
    print("\n" + "="*80)
    print("SMALL OBJECT DETAILS")
    print("="*80)

    # Group by category
    by_category = defaultdict(list)
    for obj in small_objects:
        by_category[obj['category']].append(obj)

    print(f"\n{'Category':<30} {'Count':<10} {'Min Area':<12} {'Avg Area':<12} {'Dimensions'}")
    print("-"*80)

    for cat_name in sorted(by_category.keys()):
        objects = by_category[cat_name]
        areas = [obj['area'] for obj in objects]
        min_area = min(areas)
        avg_area = sum(areas) / len(areas)
        # Get the smallest object dimensions
        smallest = min(objects, key=lambda x: x['area'])

        print(f"{cat_name:<30} {len(objects):<10} {min_area:<12.1f} {avg_area:<12.1f} {smallest['dimensions']}")

    # Show the 10 smallest objects
    print("\n" + "="*80)
    print("10 SMALLEST OBJECTS")
    print("="*80)
    print(f"\n{'Rank':<6} {'Category':<25} {'Area':<10} {'Dimensions':<15} {'ID'}")
    print("-"*80)

    for i, obj in enumerate(small_objects[:10], 1):
        print(f"{i:<6} {obj['category']:<25} {obj['area']:<10.1f} {obj['dimensions']:<15} {obj['id']}")

# Overall statistics by category
print("\n" + "="*80)
print("SIZE STATISTICS BY CATEGORY")
print("="*80)
print(f"\n{'Category':<30} {'Count':<10} {'Min Area':<12} {'Max Area':<12} {'Avg Area':<12} {'<400 px²'}")
print("-"*80)

for cat_name in sorted(size_stats.keys()):
    areas = size_stats[cat_name]
    small_count = sum(1 for a in areas if a < area_threshold)

    avg_area = sum(areas) / len(areas)
    print(f"{cat_name:<30} {len(areas):<10} {min(areas):<12.1f} {max(areas):<12.1f} {avg_area:<12.1f} {small_count:<10}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
total_small = len(small_objects)
total_annotations = len(coco_data['annotations'])
print(f"\nTotal annotations: {total_annotations}")
print(f"Objects < 400 px² (20x20): {total_small} ({100*total_small/total_annotations:.2f}%)")
print(f"Objects ≥ 400 px²: {total_annotations - total_small} ({100*(total_annotations - total_small)/total_annotations:.2f}%)")

# Check if any dimension is less than 20
very_small = [obj for obj in small_objects if obj['width'] < 20 or obj['height'] < 20]
print(f"\nObjects with width OR height < 20 pixels: {len(very_small)}")

# Objects with both dimensions < 20
tiny = [obj for obj in small_objects if obj['width'] < 20 and obj['height'] < 20]
print(f"Objects with width AND height < 20 pixels: {len(tiny)}")

if tiny:
    print("\nTiny objects (both dimensions < 20):")
    for obj in tiny[:5]:
        print(f"  - {obj['category']}: {obj['dimensions']} (area={obj['area']:.1f})")