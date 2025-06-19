import torch
import torch.nn as nn

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.w_ih = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.w_ho = nn.Parameter(torch.randn(hidden_size, vocab_size))
        self.b_o = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.xavier_uniform_(self.w_hh)
        nn.init.xavier_uniform_(self.w_ho)
    
    def forward(self, x, h):
        b_size, s_len = x.size()
        output_arr = []
        x = self.embedding(x)
        for j in range(s_len):
            x_t = x[:, j, :]
            h = torch.tanh(torch.matmul(x_t, self.w_ih) + torch.matmul(h, self.w_hh) + self.b_h)
            o_t = torch.matmul(h, self.w_ho) + self.b_o
            output_arr.append(o_t)
        output_return = torch.stack(output_arr, dim=1)
        return output_return, h
    
    def initialize_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

def load_model_and_mappings(model_path="rnn_model.pth", mappings_path="mappings.pth"):
    # Load mappings with torch.load since it was saved with torch.save
    mappings = torch.load(mappings_path, map_location=device, weights_only=False)
    character_to_index = mappings['character_to_index']
    index_to_character = mappings['index_to_character']
    vocab_size = len(character_to_index)
    hidden_size = 256  # From sp_text.py
    model = BasicRNN(vocab_size, hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, character_to_index, index_to_character

def text_generation(model, character_to_index, index_to_character, seed_text, gen_text_length=150, temperature=1.0):
    model.eval()
    h = model.initialize_hidden(1)
    input_seq = torch.tensor([character_to_index.get(ch, 0) for ch in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    generated = seed_text
    with torch.no_grad():
        for i in range(gen_text_length):
            output, h = model(input_seq, h)
            probs = torch.softmax(output[0, -1] / temperature, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            generated += index_to_character[next_char_idx]
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    return generated