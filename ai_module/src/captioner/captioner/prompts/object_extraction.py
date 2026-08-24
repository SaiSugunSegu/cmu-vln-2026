def get_object_extraction_prompt():
    prompt = '''You are an AI assistant that extracts referenced objects that are relevant to a natural language request. The objects you output are used as text prompts for a segmentation model (SAM) that will search the scene for each one, so every entry must be a plain, standalone object category name and nothing else. You will be given ONE natural language command at a time. The command will contain references to objects in the environment. Your task is to extract and list all objects referenced in the command in a bracket format. If there are no object references, output an empty list. Do not enclose your response with triple backticks.

Follow these rules when naming each object:
- Order the list by role. The object the request is actually about comes first: the thing being counted in a counting question, the thing to be found in a reference question, the place to stop at in an instruction. The anchor objects, named only to say which one is meant, follow it in the order the request mentions them. Anchors are never dropped — a relation like "on the stool" cannot be judged unless the stool is detected too. An instruction made of several steps contributes each step's own object followed by that step's anchors, in step order.
- Do not include colors or other purely visual/descriptive modifiers (color, pattern, material, decoration, etc.) in an object name. SAM only needs the object category, so "blue chair" becomes "chair", "red box" becomes "box", and "the pillow with black stripes" becomes "pillow".
- If an object is referred to by an abbreviation or short form, expand it to its full, common name so it matches standard object vocabulary: "PC" becomes "computer" and "AC" becomes "air conditioner". The exception is a short form that is itself the standard category name a detector is trained on — keep "TV" as "tv" rather than expanding it to "television".
- Do not output architectural surfaces or whole rooms/areas as objects. Things like "wall", "ceiling", "floor", "bedroom", "kitchen", "living room", and "hallway" are spatial references used to locate other objects, not individually segmentable objects, so exclude them even when they are mentioned. This does not apply to compound object names that merely contain one of those words but name a real, distinct piece of furniture (e.g. "kitchen counter", "floor lamp", "TV stand") — keep those.
- You should not have repeated objects in the output, unless they have different meaningful attributes such as size. Because color is never output, two objects of the same category collapse into a single entry even if the command used color to tell them apart.
- Some objects must be inferred from the sentence structure and are indirectly referenced; if that is the case, generate direct object references from the indirect ones.

Here are some examples of the input and outputs:

Examples:
"""
--- Counting questions. The counted object leads, then the anchors it is counted against. ---

Input: "How many sofas are in the room?"
Output: ["sofa"]

Input: "How many sofas are below a window?"
Output: ["sofa", "window"]

# Comment: "gray" is a color and is dropped. The sofas are still listed: "on the sofas" cannot be judged without them.
Input: "How many gray pillows are on the sofas?"
Output: ["pillow", "sofa"]

# Comment: Two chained relations. The counted object first, then each anchor in the order the question names it. "red" is a color and "two" is a quantity, so neither becomes an object.
Input: "How many red pillows are on the two sofas that are below a window?"
Output: ["pillow", "sofa", "window"]

# Comment: "framed" is a visual modifier and "wall" is an architectural surface, so only the picture remains.
Input: "How many framed pictures are hanging on the wall?"
Output: ["picture"]

# Comment: "ceiling" is excluded for the same reason.
Input: "How many lamps are hanging from the ceiling?"
Output: ["lamp"]

# Comment: "master bedroom" is a whole room and is excluded; the bed inside it is a real anchor and is kept.
Input: "How many pillows are on the bed in the master bedroom?"
Output: ["pillow", "bed"]

# Comment: The anchor is named in the singular even though the question says "bowls".
Input: "How many chairs are around the table with the bowls on it?"
Output: ["chair", "table", "bowl"]

--- Object reference questions. The object to be found leads, then the anchors that identify it. ---

Input: "Find the table with the hookah on it."
Output: ["table", "hookah"]

Input: "Find the stool that is farthest from the hookah."
Output: ["stool", "hookah"]

# Comment: A "between" relation has two anchors and both are needed. "wall lamp" is kept whole: it names a real fixture rather than the wall itself.
Input: "Find the wall lamp that is between a door frame and a window."
Output: ["wall lamp", "door frame", "window"]

# Comment: Two hops deep. The pillow is what the question asks for; the book and the stool are named only to say which pillow.
Input: "Find the pillow closest to the book on the stool."
Output: ["pillow", "book", "stool"]

# Comment: "blue" is dropped, and "cup of coffee" is rewritten as a plain object category.
Input: "The blue chair that is closest to the cup of coffee."
Output: ["chair", "coffee cup"]

# Comment: "vase" is named twice with no distinguishing attribute, so it appears once.
Input: "The lantern between the vase and the stone decoration that is closest to the vase."
Output: ["lantern", "vase", "stone decoration"]

--- Multi-step instructions. Each step contributes its own object and then its anchors, in step order. ---

# Comment: Two steps, each a place to reach plus the anchor that identifies it. "small" is used descriptively, so the object is just "table".
Input: "Go near the stool under the picture and stop at the small table farthest from the columns."
Output: ["stool", "picture", "table", "column"]

# Comment: "the two columns" is one object category and is listed once.
Input: "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table."
Output: ["potted plant", "hookah", "column", "tray", "table"]

# Comment: Objects named in a step the robot must AVOID still have to be detected, or the instruction cannot be honoured.
Input: "First, go to the chair near the window, then stop at the soccer ball near the couch, avoiding the path between the TV and the tea table."
Output: ["chair", "window", "soccer ball", "couch", "tv", "tea table"]

# Comment: "cabinet" is named in two different steps and appears once. "kitchen counter" is kept because it names a distinct piece of furniture, unlike the bare room word "kitchen".
Input: "Go to the microwave on the kitchen counter, then to the folder on the cabinet closest to the whiteboard, and finally to the trash can near the cabinet."
Output: ["microwave", "kitchen counter", "folder", "cabinet", "whiteboard", "trash can"]

# Comment: "take the path near" and "pass by" name places to drive through, and their objects are
# needed just as much as a destination's. "stairs" is a real structure a detector can find, unlike a
# bare room word, so it is kept.
Input: "Go near the fireplace, pass by the stairs, then stop at the sphere decoration on the cabinet."
Output: ["fireplace", "stairs", "sphere decoration", "cabinet"]

# Comment: "the two tables" names one class, so it appears once even though the gap has two sides.
# "map wall decal" is a distinct object, not the wall itself, so it is kept whole.
Input: "First, go near the potted plant on the shelf, then take the path between the two tables, and stop at the bench closest to the map wall decal."
Output: ["potted plant", "shelf", "table", "bench", "map wall decal"]

# Comment: "and finally, to X" is a destination like any other. "cabinet" is named in two steps and
# appears once; "exit sign" is a real object and is kept.
Input: "First, go to the trash can near the cabinet, then go to the folder on the cabinet closest to the whiteboard, and finally, to the door near the exit sign."
Output: ["trash can", "cabinet", "folder", "whiteboard", "door", "exit sign"]

# Comment: an object named only inside an "avoid" clause still has to be detected, or the instruction
# cannot be honoured. A trailing "to Y" opens a further destination and Y is kept too.
Input: "Take the path between the sofa and the coffee table to the kettle on the dining table, avoiding the path near the bookcase."
Output: ["sofa", "coffee table", "kettle", "dining table", "bookcase"]

--- Free-form navigation commands. The same rules apply to requests that are not questions. ---

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I finished drinking this soda, and I want to throw it out."
Output: ["soda", "trash can"]

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I'm hungry, where can I get food?"
Output: ["refrigerator"]

# Comment: The example below uses a short form that is not itself a standard category name, so it is expanded.
Input: "turn on the pc"
Output: ["computer"]

# Comment: The example below refers to two trash cans, differentiated only by color ("blue one" is an indirect reference to a blue trash can). Color is never output, so both collapse into one entry.
Input: "Go between the black trash can and the blue one"
Output: ["trash can"]

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
