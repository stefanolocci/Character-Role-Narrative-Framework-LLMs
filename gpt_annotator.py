import json
import re
from tqdm import tqdm
from openai import OpenAI
from prompts import (
    prompt_template_gpt_inline_zero_shot,
    prompt_template_gpt_inline_one_shot,
    prompt_template_gpt_inline_few_shot,
    prompt_template_gpt_inline_zero_shot_editorials,
    prompt_template_gpt_inline_one_shot_editorials,
    prompt_template_gpt_inline_few_shot_editorials,
    character_category_prompt_template,
)

# Initialize OpenAI client with the provided API key
api_key = "OPEN_AI_API_KEY"
client = OpenAI(api_key=api_key)


# -------------------- Utility Functions --------------------

def parse_annotated_text_with_original(text_to_parse, original_text):
    """
    Parses annotated text with role tags and maps it to the original text with corresponding roles.

    :param text_to_parse: Annotated text with tags like <HER>, <VIL>, etc.
    :param original_text: Original text from the dataset.
    :return: A list of annotations with text, start/end positions, and role labels.
    """
    pattern = r'<(HER|VIL|VIC|BEN)>(.*?)</\1>'
    matches = re.finditer(pattern, text_to_parse)

    ann = []
    offset = 0
    for match in matches:
        label = match.group(1)
        annotated_text = match.group(2)
        start_index_original = original_text.find(annotated_text, offset)
        end_index_original = start_index_original + len(annotated_text)
        offset = end_index_original + 1

        ann.append({
            "value": {
                "start": start_index_original,
                "end": end_index_original,
                "text": annotated_text,
                "labels": [
                    "Hero" if label == "HER" else
                    "Villain" if label == "VIL" else
                    "Victim" if label == "VIC" else
                    "Beneficiary"
                ]
            }
        })

    return ann


def parse_cc_annotated_text_with_original(text_to_parse, original_text):
    """
    Parses category annotations (Human, Instrumental, Natural) and maps them to the original text.

    :param text_to_parse: Annotated text with tags like <HUM>, <INS>, etc.
    :param original_text: Original text from the dataset.
    :return: A list of annotations with text, start/end positions, and category labels.
    """
    pattern = r'<(HUM|INS|NAT)>(.*?)</\1>'
    matches = re.finditer(pattern, text_to_parse)
    ann = []
    offset = 0
    for match in matches:
        label = match.group(1)
        annotated_text = match.group(2)
        start_index_original = original_text.find(annotated_text, offset)
        end_index_original = start_index_original + len(annotated_text)
        ann.append({
            "value": {
                "start": start_index_original,
                "end": end_index_original,
                "text": annotated_text,
                "cat": [
                    "Human" if label == "HUM" else
                    "Instrumental" if label == "INS" else
                    "Natural"
                ]
            }
        })

    return ann


# -------------------- GPT Annotation Routines --------------------

def run_gpt_annotation_CR():
    """
    Runs GPT model annotation for a specific dataset, processing each text sample
    and adding character role annotations based on the prompt template.
    """
    file_name = "dataset/NatSci_Editorials/NatSci_Final_2k.json"
    batch_name = "NatSci_Final_2k"
    final_data = []
    malformed_responses = []
    model = "gpt-4o"
    dataset = "editorials"

    # Load the dataset
    with open(file_name, "r") as file:
        data = json.load(file)

        for elem in tqdm(data):
            text = elem['Text']
            prompt = f"{prompt_template_gpt_inline_few_shot_editorials} \n{text}"

            # Generate annotation response using GPT
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=512,
                messages=[{"role": "system", "content": prompt}]
            )
            generated_content = response.choices[0].message.content
            was_cancelled = False

            try:
                annotations = parse_annotated_text_with_original(generated_content, text)
                if not annotations or not all(
                        "text" in item["value"] and "labels" in item["value"] for item in annotations):
                    was_cancelled = True
                    malformed_responses.append(generated_content)
            except Exception as e:
                was_cancelled = True
                malformed_responses.append(f"ID: -; Text: {generated_content}; Error: {str(e)}")
                annotations = []

            result = [{"value": annotation["value"]} for annotation in annotations]
            final_data.append({
                "annotations": [{"result": result, "was_cancelled": was_cancelled}],
                "data": {"Text": text},
            })

    # Save annotated data and malformed responses
    save_results(final_data, batch_name, model, dataset)
    save_malformed_responses(malformed_responses, batch_name, model, dataset)


def run_gpt_annotation_CC():
    """
    Runs GPT model for annotating category classifications (Human, Instrumental, Natural).
    Processes previously annotated data and applies category labeling.
    """
    file_name = "few-shot/editorials/gpt-4o/NatSci_Final_150_gpt-4o_annotation.json"
    batch_name = "NatSci_Final_150_CC"
    final_data = []
    malformed_responses = []
    model = "gpt-4-turbo"
    dataset = "editorials"

    # Load the dataset
    with open(file_name, "r") as file:
        data = json.load(file)

        for elem in tqdm(data):
            results = elem['annotations'][0]['result']
            text = elem['data']['Text']
            concat_res = ""

            # Annotate each entity from previous results
            for r in results:
                entity = r['value']['text']
                prompt = f"{cc_prompt_template} \n{entity}"
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    max_tokens=512,
                    messages=[{"role": "system", "content": prompt}]
                )
                concat_res += " " + response.choices[0].message.content

            try:
                annotations = parse_cc_annotated_text_with_original(concat_res, text)
                if not annotations or not all(
                        "text" in item["value"] and "cat" in item["value"] for item in annotations):
                    malformed_responses.append(concat_res)
            except Exception as e:
                malformed_responses.append(f"ID: -; Text: {concat_res}; Error: {str(e)}")
                annotations = []

            result = [{"value": annotation["value"]} for annotation in annotations]
            final_data.append({
                "annotations": [{"result": result, "was_cancelled": False}],
                "data": {"Text": text},
            })

    # Save annotated data and malformed responses
    save_results(final_data, batch_name, model, dataset)
    save_malformed_responses(malformed_responses, batch_name, model, dataset)


# -------------------- Helper Functions --------------------

def save_results(final_data, batch_name, model, dataset):
    """
    Saves the final annotated data into a JSON file.

    :param final_data: The annotated data.
    :param batch_name: The batch name used for file naming.
    :param model: The model used for annotation.
    :param dataset: The dataset being processed.
    """
    output_file = f"{model}/{dataset}/{batch_name}_{model}_annotation.json"
    with open(output_file, "w") as file:
        json.dump(final_data, file, indent=2)


def save_malformed_responses(malformed_responses, batch_name, model, dataset):
    """
    Saves malformed responses (failed annotations) to a text file.

    :param malformed_responses: List of malformed responses.
    :param batch_name: The batch name used for file naming.
    :param model: The model used for annotation.
    :param dataset: The dataset being processed.
    """
    malformed_file = f"{model}/{dataset}/{batch_name}_{model}_malformed_responses.txt"
    with open(malformed_file, "w") as file:
        for malformed in malformed_responses:
            file.write(malformed + "\n")


# -------------------- Main Script Execution --------------------

if __name__ == "__main__":
    # Run the desired annotation process (choose one)
    run_gpt_annotation_CR()  # Character role annotation
    # run_gpt_annotation_CC()  # Character category annotation
