# Protecting-Data-in-the-Age-of-Quantum-Computing-Homomorphic-Encryption-Tools-for-Machine-Learning
This repository contains tools and implementations for training and evaluating machine learning models using homomorphic encryption. By leveraging homomorphic encryption, this project ensures that data remains secure and private throughout the entire machine learning workflow, even in the face of potential quantum computing threats.

<p align="center">
  <img src="images/SCHOOL OF ENGINEERING.png" alt="College Logo" width="400">
</p>

## Project Structure

```
.
├── Encryption
│   ├── __init__.py
│   ├── EncryptData.py
│   ├── EncryptedLogisticRegression.py
│   ├── EncryptedEvaluation.py
│   ├── LogisticRegression.py
│   ├── train_model.py
│   ├── utils.py
│   └── main.py
├── data
│   ├── BreastCancer_dataset.csv
│   ├── HeartDisease_dataset.csv
│   └── Mushroom_dataset.csv
├── images
│   ├── Logo of AIML.png
|   └── SCHOOL OF ENGINEERING.png
├── requirements.txt
└── README.md
```

## Requirements

The required Python libraries are listed in the `requirements.txt` file. You can install them using the following command:

```sh
pip install -r requirements.txt
```

## Setup and Execution

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # for Linux/macOS
    .\venv\Scripts\activate   # for Windows
    ```

3. **Install the required Python libraries:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the main script:**
    ```sh
    python Encryption/main.py --path data/HeartDisease_dataset.csv --target TenYearCHD --ratio 0.2 --epochs 100 --seed 2024
    ```

    - `--path`: The path to the dataset.
    - `--target`: The target variable in the dataset.
    - `--ratio`: The ratio for the test split.
    - `--epochs`: (Optional) Number of epochs to train the model.
    - `--seed`: (Optional) Set the seed value for reproducibility.

## Note

The repository currently supports only binary classification using logistic regression. However, we have plans to extend its functionality to include other machine learning models such as Convolutional Neural Networks (CNNs), Artificial Neural Networks (ANNs), and regression models in the future. Stay tuned for updates!


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.


## Contributors

<p align="center">
  <img src="images/Logo of AIML.png" alt="College Logo" width="160">
</p>

This project is submitted as the major project by


- Akshay MR ENG20AM0007
- Anirudh Narayanan ENG20AM0010
- Chandrashekar N ENG20AM0018
- Gaurav Kamath ENG20AM0023

Students of Bachelor of Technology in Computer Science and Engineering at the School of Engineering, Dayananda Sagar University, Bangalore, in partial fulfillment for the award of a degree in Bachelor of Technology in Computer Science and Engineering(AI & ML), during the year 2023-2024.

---
