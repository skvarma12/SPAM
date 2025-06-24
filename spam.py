import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# --- Step 1: Load Dataset ---
df = pd.read_csv("train.csv")  # Rename your uploaded file to train.csv if needed
texts = df["question_text"].astype(str).tolist()
labels = df["target"].values

# --- Step 2: Tokenize and Pad Sequences ---
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(labels)

# --- Step 3: Load GloVe Embeddings ---
embeddings_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

# --- Step 4: Create Embedding Matrix ---
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# --- Step 5: Train-Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Build and Compile the Model ---
model = Sequential([
    Embedding(input_dim=num_words,
              output_dim=EMBEDDING_DIM,
              weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH,
              trainable=False),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# --- Step 7: Train the Model ---
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

# --- Step 8: Evaluate and Save ---
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")

model.save("quora_spam_model.keras")
print("Model saved to quora_spam_model.keras")

with open("model_config.json", "w") as f:
    f.write(model.to_json())
print("Model architecture saved to model_config.json")
