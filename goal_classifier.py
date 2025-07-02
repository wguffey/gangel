import os
import json
from google import genai

client = genai.Client()


def get_examples():
    return {
        "example_1": {
            "input": {
                "goals": [
                    "Learn about bunnies",
                    "Learn about gardening",
                ],
                "content": "spaceship blows up during launch in texas",
            },
            "output": [],
        },
        "example_2": {
            "input": {
                "goals": [
                    "Learn about bunnies",
                    "Learn about gardening",
                ],
                "content": "Scientist finds bunnies help garden grow",
            },
            "output": ["Learn about bunnies", "Learn about gardening"],
        },
    }


# def classify_content_with_gemini(goals, content):
def classify_content_with_gemini(goals, content):
    input_ = {
        "goals": goals,
        "content": content,
    }

    prompt = f"""
You are an AI classifier tasked with figuring out if content aligns with specified user goals.
The broader context is that a user wants to only see content on the internet that aligns with their specified goals.
You are the classifier that says if content is aligned with stated goals or not.
You are given a set of goals and some content and need to respond with a list of the goals that the content aligns with.

Examples:
{json.dumps(get_examples()).replace("{", "{{").replace("}", "}}").strip()}

Now time for you to do this classification. Below are the goals and content to be used during classification:
{input_}
"""
    print("prompt start:")
    print(prompt)
    print("prompt end:")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error during API call: {e}"


def load_tests():
    filepath = "tests.jsonl"
    tests = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    tests.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()} - {e}")

    return tests


def main():
    """
    Main function to run the goal classifier with predefined goals and test cases from a file.
    """
    # Define your goals here
    goals = [
        "learn machine learning",
        "stay up to date on AI",
        "read about startups",
        "improve python programming skills",
    ]

    # Read test cases from file
    # try:
    #     with open("test_cases.txt", "r") as f:
    #         test_cases = [
    #             line.strip() for line in f.readlines() if line.strip()
    #         ]
    # except FileNotFoundError:
    #     print("Error: test_cases.txt not found. Please create this file.")
    #     return
    tests = load_tests()

    # print("--- Goal-Content Aligner ---")
    # print("Goals:", goals)
    # print("------------------------------------------")

    for i, test in enumerate(test_cases):
        goals = test["goals"]
        content = test["content"]
        ground_truth = test["alignment"]
        classification = classify_content_with_gemini(goals, content)
        print(f'Test Case {i+1}: "{test}"')
        print(f"  -> prediction: {classification}")
        print(f"  -> ground truth: {ground_truth}")
        print("------------------------------------------")
        break


if __name__ == "__main__":
    main()
