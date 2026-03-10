import json
import random
import argparse
from prompt_versions import prompt_v1, prompt_v2, prompt_v3, prompt_v4

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default="socialiqa_results_20260304_111247.jsonl")
parser.add_argument("--output_file", default="training_data.jsonl")
args = parser.parse_args()

with open(args.input_file, "r") as f_in, \
     open(args.output_file, "w") as f_out:
    for line in f_in:
        record = json.loads(line)
        if record['correct']:
            input_data = record["input"]
            context = input_data["context"]
            question = input_data["question"]
            q_options = []
            for option in input_data.keys():
                if option.startswith("answer"):
                    q_options.append(f"{option}: {input_data[option]}")
            answer = record["gold_answer"]
            final_response = record["final_response"]
            hypothesis_dict = record["selected_hypothesis"]
            hypothesis = hypothesis_dict["explanation"]
            if hypothesis.startswith("(Refined by DomainAgent)"):
                hypothesis = hypothesis[len("(Refined by DomainAgent) "):]
            hypothesis_type = hypothesis_dict["type"]

            prompt_versions = [prompt_v1, prompt_v2, prompt_v3, prompt_v4]
            chosen_prompt = random.choice(prompt_versions)
            sys_prompt, prompt = chosen_prompt(context, question, q_options)

            datapoint = {
                "system": sys_prompt,
                "prompt": prompt,
                "response": {
                    "Mental State Type": hypothesis_type,
                    "Hypothesis": hypothesis,
                    "Response": final_response,
                    "Final answer": answer
                }
            }
            f_out.write(json.dumps(datapoint) + "\n")
