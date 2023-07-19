# Identification-of-features-from-books

The code in these repository perfrom the following feature extraction for book. The task is to generate table of content, set of questions, set of character names and set of named entities (with types) from a UTF-8 plain text serialized book.

The books are sourced from e https://www.gutenberg.org to create the training set and evaluatio set.

task1 – Use regular expressions (regex) to capture chapter titles from the book's content and create a JSON file containing the table of contents. Save this file in the present working directory. In case the book already possesses its own table of contents, disregard it, and generate a new one solely from the text's extracted chapter headings.

task2 - Apply regular expressions (regex) to extract each question present in the chapter text. Serialize these questions into a text file and save it to the current working directory.

task3 - Create and train a Conditional Random Field (CRF) model to identify and tag named entities using the ontonotes dataset as the training data. Execute the CRF model on the chapter text and serialize the recognized named entities into a JSON format. Only include named entities of specific types, such as DATE, CARDINAL, ORDINAL, and NORP. Save this JSON file containing the extracted named entities to the current working directory.

task4 – Utilize both the CRF model and regular expressions (regex) to extract all character names from the chapter text. Serialize these extracted character names into a text file and save it in the current working directory.
