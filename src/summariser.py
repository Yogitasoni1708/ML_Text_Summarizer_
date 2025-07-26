from transformers import pipeline

# Step 1: Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 2: Read input text from file
with open("ML_TEXT_SUMMARIZATION.py/data/input.txt", "r", encoding="utf-8") as file:

    input_text = file.read()

# Step 3: Generate summary
summary = summarizer(input_text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

# Step 4: Display result
print("\nOriginal Text:\n", input_text)
print("\nGenerated Summary:\n", summary)
