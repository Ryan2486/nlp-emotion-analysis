# NLP Emotion Analysis
This project is a hands-on exploration of emotion detection using transformer-based models, specifically [DistilBERT](https://huggingface.co/distilbert-base-uncased). The entire code is written in a single script to keep the flow linear, easy to follow, to allow a progressive, experimental flow where we explore tokenization, encoding, dimensionality reduction, classification, and fine-tuning.

The code includes `print()` statements to indicate progress, but **all explanations, reflections, and observations are here — in this README**.

## What You'll Find Here

- Testing the `distilbert-base-uncased` tokenizer and encoder
- Visualizing sentence embeddings using UMAP
- Training a simple Logistic Regression classifier
- Fine-tuning a pre-trained DistilBERT model on the [HuggingFace "emotion" dataset](https://huggingface.co/datasets/dair-ai/emotion)
- Evaluating performance using confusion matrix and accuracy
- Step-by-step guidance inside the code for reproducibility

The code is structured to be followed line by line. At each step.

## Purpose
This project isn't meant to be production-ready. Instead, it's a notebook-style journey to understand how modern NLP models work when applied to emotional text classification — with a touch of visual intuition and classic ML comparison.

## So...Let's Get Started 😎
It's easy to load a sample dataset using 🤗(Hugging Face) Datasets — just one line of code!

In our case, we use the `emotion` dataset, which contains short text messages labeled with emotional categories like joy, anger, sadness, and more.

### Code:
```python
print("Loading a dataset from Hugging Face Datasets")
from datasets import load_dataset

emotions = load_dataset("emotion")
```
This dataset is great for exploration: it’s small enough to experiment quickly, but rich enough to visualize patterns and train models that make meaningful predictions.
>The first time you run this, 🤗 will download the dataset and store it in your local cache (usually under `~/.cache/huggingface/`).
The next time, it loads instantly from cache — no internet required!

Once the dataset is loaded, you can, for example, check the size of it using `len`.
```python
print(f"Length of the dataset: {len(emotions)}")
```
And you can have something like this:
```
Length of the dataset: 3
```
That’s because most NLP datasets are usually split into **three** parts:
- Train: used to teach the model,
- Validation: used during training to evaluate how well the model is generalizing,
- Test: used at the very end to measure the final performance.

Why usually train, validation, and test?🤔
> Because if you train and evaluate on the same data, you're basically asking the model to recite, not to understand. We want to test if it actually **generalizes** — not just memorizes.

So, let’s check the name of the columns with `column_names`:
```python
print("Checking the structure of the dataset:", emotions["train"].column_names)
```
```
Checking the structure of the dataset: ['text', 'label']
```
This shows that each example in the dataset has two columns:
- `text`: the actual text message (like "I am so happy today!"),
- `label`: the corresponding emotion label (like "joy").

But we can go a bit deeper and explore what these columns actually contain using `.features` :
```python
print("Exploring the dataset features:", emotions["train"].features)
```
```
Exploring the dataset features: {'text': Value(dtype='string', id=None), 
'label': ClassLabel(num_classes=6, names=['joy', 'sadness', 'anger', 'fear', 'surprise', 'love'], id=None)}
```
In this case, the data type of the `text` column is ``string``, while the ``label`` column is a
special ``ClassLabel`` object that contains information about the class names and their
mapping to integers.

Now let’s dive into a classic data science habit: turning everything into a DataFrame — because if it’s not a DataFrame, did you even analyze it? 😏
```python
print("Importing pandas for DataFrame manipulation")
import pandas as pd

emotions.set_format(type="pandas")
```
This line tells 🤗 datasets to give us the data in pandas format, so we can work with it like pros (and avoid looping through dictionaries like cavemen).

After that, we can grab the training set and convert it into a DataFrame:
```python
df = emotions["train"][:]
```
DataFrame is basically a fancy table with superpowers 😋.

So... let's take a peek at what we’re working with 👀.
```python
print("Checking the structure of the DataFrame:")
print(df.head())
```
```
Checking the structure of the DataFrame:
                                                text  label
0                            i didnt feel humiliated      0
1  i can go from feeling so hopeless to so damned...      0
2   im grabbing a minute to post i feel greedy wrong      3
3  i am ever feeling nostalgic about the fireplac...      2
4                               i am feeling grouchy      3
```
Numbers are great for computers, but we humans prefer words.

So, we’ll translate the numeric label into a readable string like "joy", "anger", etc.

To do that, we’ll define a simple function:
```python
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)
```
``emotions["train"].features["label"]`` is a ClassLabel object.
This special class has a method called ``.int2str()`` which converts an integer label (e.g., ``2``) into a string label (e.g., ``joy``).

Now let’s apply it to every row:
```python
df["label_name"] = df["label"].apply(label_int2str)
```
This creates a new column called "label_name" with readable labels. 🎉

Let's check the updated DataFrame:
```python
print("Checking the new structure of the DataFrame: ")
print(df.head())
```
```
                                                text  label label_name
0                            i didnt feel humiliated      0    sadness
1  i can go from feeling so hopeless to so damned...      0    sadness
2   im grabbing a minute to post i feel greedy wrong      3      anger
3  i am ever feeling nostalgic about the fireplac...      2       love
4                               i am feeling grouchy      3      anger

```
Much better, right? 😏

Now that we’ve got our label_name column with readable emotions instead of just numbers, 
let’s visualize how balanced (or unbalanced) the dataset is — 
because not all emotions are equally represented… and that’s okay, life isn’t fair either 😅
```python
print("Visualizing the distribution of labels in the dataset")
import matplotlib.pyplot as plt

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
```
``.plot.barh()`` makes a horizontal bar chart. Why horizontal? Because it’s prettier. Just kidding... it often fits labels better

Look at that! We can see the distribution of emotions in our dataset.

![Frequency of Classes](./Image/Frequency)

You get a nice bar chart showing how emotions like joy, anger, or fear are spread across the training dataset. Great way to spot imbalances — which matter a lot for model training. For example:
there's 8000 "joy" samples but only 800 "surprise," our model might become a joy-addict. 😂

Want to fix that? Techniques like **oversampling**, **undersampling**, or **class weights** can help — but we’ll keep that for another time 😉

Next, we execute this:
```python
print("Resetting the format of the dataset to its original structure")
emotions.reset_format()
```
We’re using ``reset_format()`` to undo any temporary formatting (like converting it to pandas).
But honestly…
>**We could’ve skipped it entirely.** 
> 
>We just wanted to feel fancy and play with DataFrames for better visuals and manipulation. 🤓

So, ``emotions`` is now once again in its raw, native, ready-to-use 🤗 format!

---
Just like loading a dataset with 🤗 is easy 😊, grabbing a model—or a part of it, like its tokenizer—is just as simple 😎.
Luckily, I’m living in this era where all these powerful tools are just a few lines of code away 😅!

>We’re not going to reinvent the wheel by coding everything from scratch—like the tokenizer and other core components. The goal here is to understand how a model works in general before trying to build one ourselves.

We now load the tokenizer associated with the transformer model `distilbert-base-uncased`.

Tokenization is the step where raw text is converted into a sequence of token IDs that the model can understand. These tokens are often subwords (like "play" + "##ing").

You can think of the tokenizer as the language of your model — without it, it simply won’t understand what you’re trying to say😝.

### Code:
```python
print("Loading a pre-trained tokenizer from Hugging Face Transformers")
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print("Tokenizer loaded successfully!")
```

>Just like the dataset, Hugging Face will download the tokenizer files the first time and store them in your local cache (`~/.cache/huggingface/transformers/`). Subsequent runs will reuse these files — no re-download needed.
>This is true for almost everything you load from 🤗 — whether it's a dataset, a tokenizer, or a pre-trained model, there will always be an initial download the first time.

>The `distilbert-base-uncased` tokenizer is case-insensitive — "Hello" and "hello" will be treated the same.

---
After loading the tokenizer, let’s try it out on a simple example.
### Code:
```python
print("Tokenizing a sample text using the pre-trained tokenizer")
text = "Tokenizing text is a core task of NLP"
print("text:", text)
encoded_text = tokenizer(text)
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
```

Here, my print() statements show:
- The original text we want to tokenize

- The raw output from the tokenizer (includes input_ids, attention_mask, etc.)

- The actual tokens (subwords) that correspond to each ID

### Output: 
```
Tokenizing a sample text using the pre-trained tokenizer
text: Tokenizing text is a core task of NLP
{'input_ids': [101, 19204, 3793, 3793, 2003, 1037, 4569, 4708, 1997, 17953, 102], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '[SEP]']
```
We get a dictionary called encoded_text, which includes:

- `input_ids`: the numerical representation of each token (word or subword)

- `attention_mask`: a very useful feature that tells the model which tokens to pay attention to and which to ignore

I'll explain `attention_mask` in more detail later — but for now, just know it's important, especially when dealing with padding or variable-length sequences.

>Notice how the word "Tokenizing" is split into 'token' and '##izing' — this is a subword tokenization strategy called WordPiece, used by BERT-based models.
The special tokens [CLS] and [SEP] are automatically added: [CLS] marks the start of the input, and [SEP] is used to separate segments.

---
So, that’s good — we know how to tokenize a single sentence.

But now, we need to tokenize every sentence in our dataset. And one thing that’s really important here… the length!

Why🤨? Transformer models like `DistilBERT` have a maximum input length (typically 512 tokens).
If a sentence is longer, it will be truncated, meaning some information will be lost. If it’s shorter, it will be padded so all inputs in a batch have the same size — this is necessary for efficient computation.

Getting padding and truncation right is crucial to avoid unexpected behavior during training and inference.

To ensure every sentence gets tokenized properly, we can give our `tokenizer` a batch of sentences and tell it:
>“Hey, if some sentences are shorter than the others, add padding to match. And if some are too long, feel free to truncate them.”

### Code:
```python
print("Define a function to tokenize a batch of texts")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

print("Tokenizing the first two samples from the training set")
print(tokenize(emotions["train"][:2]))
```

Here, my `print()` statements show the output of tokenizing the first two training samples as a batch

### Output:
```
Define a function to tokenize a batch of texts
Tokenizing the first two samples from the training set
{'input_ids': [[101, 1045, 2134, 2102, 2514, 26608, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [101, 1045, 2064, 2175, 2013, 3110, 2061, 20625, 2000, 2061, 9636, 17772, 2074, 2013, 2108, 2105, 2619, 2040, 14977, 1998, 2003, 8300, 102]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```
Like we can see, the length of our two sentences is not the same.
So, we add 0s to the attention mask to tell the future model: 
>"Ignore these parts, they are just padding."😗

Now you understand the role of the attention mask!👏

---

Okay so, now that we understand what tokenization is, and how `input_ids` and `attention_mask` work...

We are ready to tokenize the entire training set!
```python
print("Tokenizing the entire training set")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print("Tokenization complete!")
```
This will apply our `tokenize` function to every example in the dataset, all at once.
Thanks to `batched=True`, it processes the data in batches, which is more efficient 😉.

Let’s check the structure now with
```python
print("Checking the structure of the tokenized dataset :", emotions_encoded["train"].column_names)
```
```
Checking the structure of the tokenized dataset : ['text', 'label', 'input_ids', 'attention_mask']
```
You'll now see that each example contains `input_ids`, `attention_mask`, and the original labels.
And There! 🎉 Our dataset is fully tokenized and ready to be used with a Transformer model

---
Now that we have a nicely tokenized dataset, it's time to bring in the real MVP...

The pre-trained Transformer model.
```python
print("Loading a pre-trained encoder/model from Hugging Face Transformers")
import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
print("Encoder/model loaded successfully!")
```

So what’s going on here?
- `torch.device(...)` → We check if we have a GPU (`CUDA`). If yes, we use it. If not... CPU it is.

- `AutoModel.from_pretrained(...)` → This line loads a pre-trained encoder (like `DistilBERT`, `BERT`, `RoBERTa` etc.) from 🤗 using the model checkpoint we defined earlier.

- `.to(device)` → We make sure our model goes to the same device (GPU/CPU) as our data.

The model is now ready to process tokenized inputs like a boss.

In our case, we use `DistilBERT`, which is an encoder-only Transformer.
This means we’re only interested in understanding the meaning of the sentence —
not generating new text.

The tokenizer helps split and encode the text into tokens (input IDs),
but it's the encoder (`DistilBERT`) that gives us a deeper semantic representation
of the sentence.

Think of it like this:

>The tokenizer does the surface work: splitting and encoding.
>
>The encoder does the deep thinking: what do these tokens mean together?

Note that :
>When using a pre-trained model, it is not just important — it's essential to use the same tokenizer that was used during its pretraining.
>
>Why?
>Because the model has learned to associate specific token IDs with specific word patterns based on that tokenizer.
>
>If you change the tokenizer, you’re basically changing the model’s entire language —
>like saying “cat” when you mean “bed”.

The model will still process the input, but it’ll make wrong associations —
and your performance will collapse.

Using the original tokenizer ensures the input IDs line up correctly
with what the model expects internally.
