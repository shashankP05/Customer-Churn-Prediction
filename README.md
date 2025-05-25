# Customer Churn Prediction

A simple machine learning project that predicts which customers might cancel their subscription using artificial intelligence.

## What does this do?

This project helps businesses understand which customers are likely to leave so they can take action to keep them. It's like having a crystal ball that warns you when a customer might cancel!

## The Data

We use a dataset called `Churn.csv` with information about 7,044 customers. It includes things like:

- How old they are and if they're married
- How long they've been a customer
- What services they use (phone, internet, etc.)
- How much they pay each month
- Whether they actually left or stayed

## How it Works

The program uses a "neural network" (a type of AI) that learns patterns from customer data. It's like training a computer to recognize signs that someone might leave.

The AI has 3 layers:
- First layer: Looks at all customer information
- Middle layers: Finds hidden patterns
- Final layer: Makes a Yes/No prediction about leaving

## How to Use

### What you need first:
```bash
pip install pandas scikit-learn tensorflow
```

### Run the code:

1. Put the `Churn.csv` file in your folder
2. Run this code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('Churn.csv')

# Prepare the data for AI
x = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create the AI model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(x_train.columns)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Train the AI
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=32)

# Test how well it works
predictions = model.predict(x_test)
predictions = [0 if val < 0.5 else 1 for val in predictions]
accuracy = accuracy_score(y_test, predictions)
print(f'The AI is {accuracy:.1%} accurate!')

# Save the trained model
model.save('tfmodel.keras')
```

## What happens?

1. The program reads customer data
2. It learns patterns from 80% of the data
3. It tests itself on the remaining 20%
4. It tells you how accurate it is
5. It saves the trained AI for future use

## Files you'll have:

- `Churn.csv` - Your customer data
- `tfmodel.keras` - Your trained AI model
- Your Python script

## Why is this useful?

- **Spot trouble early**: Know which customers might leave
- **Save money**: Keep customers instead of finding new ones
- **Better business**: Focus on customers who need attention
- **Smart decisions**: Use data instead of guessing

## Making it better:

- Try different AI settings
- Add more customer information
- Test with your own data
- Make it predict faster

## Need help?

If something doesn't work:
1. Make sure all packages are installed
2. Check that `Churn.csv` is in the right folder
3. Make sure your data has the right column names
