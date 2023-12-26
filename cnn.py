# Convolutional Neural Network
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('/home/zihad/Lab/TheCsvFIle.csv', encoding =('ISO-8859-1'),low_memory =False)
#df['class'] = df['class'].replace('non-suicide', 0)
#df['class'] = df['class'].replace('suicide', 1)
df = df.rename(columns = {"message to examine":"text", "label (depression result)" :"label" })
df = df[["text", "label"]]
# Tokenize and pad sequences
max_words = 10000  # Adjust based on your dataset size
max_len = 100  # Adjust based on your sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df['label'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build the CNN model
embedding_dim = 50  # Adjust based on your word embeddings
filters = 128
kernel_size = 3
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
batch_size = 64
epochs = 5

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy}")

