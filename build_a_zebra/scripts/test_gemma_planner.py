import os
from PIL import Image
from google import genai
from secrets import GOOGLE_API_KEY


client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])#print("Available models:")
#print("Available models:")
#for m in client.models.list():
#    print(" -", m.name)

image = Image.open("test_scene.png")   # ← make sure this file exists

prompt = ("""
You must ONLY use the exact part IDs listed below.
You must NOT invent, modify, or abbreviate any ID.
If you invent an ID, the answer is WRONG.

Allowed part IDs (copy exactly):
- 31111p0e
- 31111p0f
- 31111p0g

Available skills: PICK, PLACE, FLIP.

Current state:
- 31111p0e (legs): on the table
- 31111p0f (body): not yet placed
- 31111p0g (head): not yet placed

Look at the image. Answer with EXACTLY one of these forms:
  PICK 31111p0e
  PICK 31111p0f
  PICK 31111p0g
  PLACE 31111p0e
  PLACE 31111p0f
  PLACE 31111p0g
  FLIP 31111p0e
  FLIP 31111p0f
  FLIP 31111p0g
  DONE

Any other answer is wrong.
"""
)

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents=[image, prompt],
)

print("\n--- Gemma's Decision ---")
print(response.text)