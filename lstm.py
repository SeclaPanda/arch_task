import torch
import torch.nn as nn

# Prepare the data
text = "Hello, world!"
chars = list(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = dict(enumerate(chars))
data = [char_to_idx[char] for char in text]

# Convert the data into batches
seq_length = 4
inputs = []
targets = []
for i in range(len(data) - seq_length):
    inputs.append(data[i:i+seq_length])
    targets.append(data[i+seq_length])
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# Define the model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = len(chars)
hidden_size = 128
output_size = len(chars)
model = CharLSTM(input_size, hidden_size, output_size)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")
