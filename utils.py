from transformers import AutoModel, AutoTokenizer
from GCN import GCN
import torch


def _norm(s):
    return ' '.join(s.strip().split())


def build_model(model_name="./blenderbot_small-90M", input_size=256, output_size=9, tokenizer_only=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer_only:
        return tokenizer
    model = GCN(input_size, output_size)
    return tokenizer, model


def utt_encoder(txt, tokenizer):
    txt = _norm(txt)
    encoded = torch.tensor(tokenizer.encode(
        txt, padding='max_length', truncation=True, max_length=128))
    return encoded


def convert_data_to_inputs(data, tokenizer: AutoTokenizer, strategy2id):
    def process(x): return tokenizer.encode(
        x, padding='max_length', truncation=True, max_length=256)

    dialog = data['dialog']
    inputs = []
    labels = []
    # content = []
    # current_speaker = None

    for i in range(len(dialog)):
        # normalize raw texts (remove space, etc.)
        text = _norm(dialog[i]['content'])
        text = process(text)
        speaker = dialog[i]['speaker']

        # if current_speaker is None:
        #     current_speaker = speaker
        # if speaker == current_speaker:
        #     content.append(text)

        # if speaker != current_speaker or i == len(dialog) - 1:
        #     res = {
        #         "content": sum(content, []),
        #         "speaker": current_speaker,
        #     }

        #     inputs.append(res)
        #     current_speaker = speaker
        #     content = [text]

        res = {
            'content': text,
            'speaker': speaker,
        }
        inputs.append(res)
        if speaker == "supporter":
            label = dialog[i]['annotation']['strategy']
            labels.append(strategy2id['['+label+']'])
        else:
            labels.append(8)

    return inputs, labels


def featurize(nodes):
    feature = [node.get_encoded() for node in nodes]
    return torch.tensor(feature, dtype=torch.float)
