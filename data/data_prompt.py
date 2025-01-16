import tiktoken, warnings
from transformers import AutoTokenizer, AutoConfig

class BasePromptTemplate:
    def __init__(self, config):
        self.config = config
        self.is_openai = config["framework"] == "openai"
        self.max_input_len = config['generator_max_input_len']
        if not self.is_openai:
            self.generator_path = config["generator_model_path"]
            model_config = AutoConfig.from_pretrained(self.generator_path, trust_remote_code=True)
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if "chat" in model_name or "instruct" in model_name:
                self.is_chat = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
        else:
            self.is_chat = True
            self.enable_chat = True
            try:
                self.tokenizer = tiktoken.encoding_for_model(config['generator_model'])
            except Exception as e:
                print("Error: ", e)
                warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
                self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    def truncate_prompt(self, prompt):
        if self.is_openai:
            truncated_messages = []
            total_tokens = 0
            assert isinstance(prompt, list)
            for message in prompt:
                role_content = message['content']
                encoded_message = self.tokenizer.encode(role_content)

                if total_tokens + len(encoded_message) <= self.max_input_len:
                    truncated_messages.append(message)
                    total_tokens += len(encoded_message)
                else:
                    print(f"The input text length is greater than the maximum length ({total_tokens + len(encoded_message)} > {self.max_input_len}) and has been truncated!")
                    remaining_tokens = self.max_input_len - total_tokens
                    truncated_message = self.encoding.decode(encoded_message[:remaining_tokens])
                    message['content'] = truncated_message
                    truncated_messages.append(message)
                    break

            return truncated_messages

        else:
            assert isinstance(prompt, str)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > self.max_input_len:
                print(f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!")
                half = int(self.max_input_len / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                        self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt

class WrongAnswerPromptTemplate(BasePromptTemplate):
    placeholders = ["answer", "reference", "question"]
    base_system_prompt = (
        "Based on a given question and its correct answer, generate a misleading wrong answer."
        "You can refer to some relevant documents for inspiration." 
        "The wrong answer should belong to the same entity type as the correct answer (e.g., person, time, place, organization, data, etc.) to enhance its confusion."
        "If the answer does not contain an entity, replace a key entity in the question and treat it as the wrong answer."
        "Only give me the wrong answer and do not output any other words."
        "\nThe following are the question and relevant documents.\n\n{reference}\n"
        "Question: {question}"
    )
    base_user_prompt = "Correct Answer: {answer}"

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):

        super().__init__(config)

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
    
    def format_reference(self, passages):
        format_reference = ""
        for idx, content in enumerate(passages):
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def get_string(self, question=None, passages=None, answer=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        input_params = {"question": question, "reference": self.format_reference(passages), "answer": str(answer)}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if not self.is_openai:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])


        return self.truncate_prompt(input)
    
class CounterFactualPassagePromptTemplate(BasePromptTemplate):
    placeholders = ["wrong_answer", "answer", "passage"]
    base_system_prompt = (
        "You are a writing AI."
        "Rewrite the passage by replacing all content and information related to {answer} with {wrong_answer}."
        "Ensure that the rewritten passage is fluent and concise, maintaining a language style similar to the original."
        "Only give me the rewritten passage and do not output any other words."
    )
    base_user_prompt = "Original Passage: {passage}\n"

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):

        super().__init__(config)

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template

    def get_string(self, passage=None, answer=None, wrong_answer=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)


        input_params = {"passage": passage, "answer": str(answer), "wrong_answer":wrong_answer}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if not self.is_openai:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])


        return self.truncate_prompt(input)
