import json

def food_tool():
    """Useful to answer open ended questions about food by giving a list of food resources.
    Returns a list of food resources"""
    entries = []
    with open('CategorizedData2.jsonl', 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            if entry.get('general_category') == 'Food':
                entries.append(entry)
    return entries

print(food_tool())