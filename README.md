# Two Tower Retrieval Recommendation System

## Overview

The Two Tower Retrieval Recommendation System revolutionizes personalized recommendations by efficiently matching user preferences with relevant items. By leveraging separate neural networks for users and items, this system ensures accuracy and effectiveness in recommendation matching. This README provides a comprehensive overview of the project, including its architecture, training process, and usage.
![Tower Architecture](img/two-tower-model-architecture-implemented.png)
## Architecture

The Two Tower Retrieval Recommendation System comprises two distinct neural networks: one dedicated to users and the other to items. Each network processes entity-specific features and generates embeddings, representing users and items in a high-dimensional space. This architecture allows the system to capture intricate user preferences and detailed item characteristics for precise recommendation matching.

![Two Tower Architecture](img/model.png)
## Online Retrieval
During online retrieval, the query tower dynamically generates a user embedding using real-time user-side features. This embedding serves as a representation of the user's preferences and characteristics. The system then utilizes an online approximate nearest neighbors (ANN) search service, such as Annoy search, to efficiently identify the most relevant items for the given user.

By leveraging the user embedding and ANN search, the system can quickly retrieve a subset of items that are likely to be of interest to the user. This process enables the system to deliver timely and accurate recommendations, enhancing the user experience and increasing engagement.
![Offline Architecture](img/Instagram-Explore-Ranking_image6.webp)
## Files Included

1. `ttrecsys.ipynb`: Jupyter notebook for training the Two Tower model and evaluating the trained model's performance.
2. `README.md`: Documentation file explaining the project and providing instructions on how to use it effectively.

## Dependencies

This project relies on the following Python libraries:

- TensorFlow: Deep learning framework for building and training neural networks.
- Keras: High-level neural networks API for easy model building and training.
- NumPy: Library for numerical computations and array manipulation.
- Pandas: Data manipulation and analysis library for handling datasets.
- Hugging Face Transformer: Library for natural language processing tasks and transformer-based models.
- Annoy: Approximate nearest neighbors implementation for efficient retrieval of similar items.

Install these dependencies using `pip`:

```
pip install tensorflow keras numpy pandas transformers annoy
```

## Usage

To use the Two Tower Retrieval Recommendation System, follow these steps:

1. Prepare your dataset with user interactions and item attributes.
2. Train the Two Tower model using the provided Jupyter notebook (`ttrecsys.ipynb`).
3. Evaluate the trained model's performance and fine-tune hyperparameters as needed.
4. Deploy the trained model for online retrieval, ensuring to generate user embeddings dynamically and utilize an approximate nearest neighbors (ANN) search service for efficient item retrieval.

Experiment with different hyperparameters, data preprocessing techniques, and model architectures to optimize the system's performance further.

## Author

Kaustubh Gupta
