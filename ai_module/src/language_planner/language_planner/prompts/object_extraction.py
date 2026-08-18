def get_object_extraction_prompt():
    prompt = '''You are an AI assistant that extracts referenced objects that are relevant to a natural language request. The objects you output are used as text prompts for a segmentation model (SAM) that will search the scene for each one, so every entry must be a plain, standalone object category name and nothing else. You will be given ONE natural language command at a time. The command will contain references to objects in the environment. Your task is to extract and list all objects referenced in the command in a bracket format. If there are no object references, output an empty list. Do not enclose your response with triple backticks.

Follow these rules when naming each object:
- Do not include colors or other purely visual/descriptive modifiers (color, pattern, material, decoration, etc.) in an object name. SAM only needs the object category, so "blue chair" becomes "chair", "red box" becomes "box", and "the pillow with black stripes" becomes "pillow".
- If an object is referred to by an abbreviation or short form, expand it to its full, common name so it matches standard object vocabulary. For example, "TV" becomes "television" and "PC" becomes "computer".
- Do not output architectural surfaces or whole rooms/areas as objects. Things like "wall", "ceiling", "floor", "bedroom", "kitchen", "living room", and "hallway" are spatial references used to locate other objects, not individually segmentable objects, so exclude them even when they are mentioned. This does not apply to compound object names that merely contain one of those words but name a real, distinct piece of furniture (e.g. "kitchen counter", "floor lamp", "TV stand") — keep those.
- You should not have repeated objects in the output, unless they have different meaningful attributes such as size. Because color is never output, two objects of the same category collapse into a single entry even if the command used color to tell them apart.
- Some objects must be inferred from the sentence structure and are indirectly referenced; if that is the case, generate direct object references from the indirect ones.

Here are some examples of the input and outputs:

Examples:
"""
Input: "the pillow with black stripes near the couch"
Output: ["pillow", "couch"]

Input: "can you get me my coffee cup on the kitchen counter"
Output: ["coffee cup", "kitchen counter"]

Input: "find me something to eat"
Output: ["something to eat"]

Input: "move the red box between the chair and the desk"
Output: ["box", "chair", "desk"]

# Comment: The example below has a repeated "chair", but "other" is not a different attribute, so we only output "chair".
Input: "move the chair between the other chair and the desk"
Output: ["chair", "desk"]

# Comment: The example below has a repeated "chair", differentiated only by color in the input. Color is never output, so both collapse into one entry.
Input: "the chair that is in between the red chair and the book",
output: ["chair", "book"]

# Comment: The example below has a repeated "monitor", but they have the same attributes.
Input: "the monitor that is in the middle of both of the monitors",
output: ["monitor"]

# Comment: The example below has a repeated "pillow", differentiated only by the color of the heart on each. Color is never output, so both collapse into one entry.
Input: "The pillow between the pillow with a black heart on it and the pillow with a red heart on it"
Output: ["pillow"]

# Comment: The example below refers to two trash cans, differentiated only by color ("blue one" is an indirect reference to a blue trash can). Color is never output, so both collapse into one entry.
Input: "Go between the black trash can and the blue one"
Output: ["trash can"]

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I finished drinking this soda, and I want to throw it out."
Output: ["soda", "trash can"]

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I'm hungry, where can I get food?"
Output: ["fridge"]

# Comment: The example below uses a short form/abbreviation, which must be expanded to its full common name.
Input: "turn on the tv"
Output: ["television"]

# Comment: The example below mentions "wall", an architectural surface, which is excluded even though it is referenced.
Input: "the picture frame hanging on the wall"
Output: ["picture frame"]

# Comment: The example below mentions "bedroom", a whole room, which is excluded; "nightstand" is a real object and is kept.
Input: "walk into the bedroom and grab my keys from the nightstand"
Output: ["keys", "nightstand"]

# Comment: The example below mentions "ceiling", an architectural surface, which is excluded. "floor lamp" is kept because it names a real object, not the floor itself.
Input: "the light fixture hanging from the ceiling above the floor lamp"
Output: ["light fixture", "floor lamp"]

"""
End Examples
'''
    return prompt


def get_obj_retrieval_prompt():
    prompt = '''You are an AI assistant that retrieves relevant objects of the same type from a scene given a target list. You will be given a list of target objects and a dictionary of scene objects. Please return the ids of all the objects from the scene that are mentioned in the target objects or of the same type. ONLY return a list of integer object IDs in your response. Do not return ANYTHING ELSE. Make sure you get the mentioned object. Here are some examples of the input and outputs:

Examples:
"""
Targets=["black chair", "window"], 
Scene objects={"0":"chair", "1":"couch", "2":"chair", "3":chair", "4":"table", "5":"microwave", "6":"pillow", "7":"window"}
Output: 
["0", "2", "3", "7"]

Targets=["coffee cup", "sofa"], 
Scene objects={"0":"chair", "1":"couch", "2":"chair", "3":cup", "4":"table", "5":"microwave", "6":"pillow", "7":"window", "8":"couch"}
Output: 
["1", "3", "8"]
"""

'''
    return prompt
